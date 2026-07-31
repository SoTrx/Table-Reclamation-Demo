import json
import re
import time
from pathlib import Path

import dotenv
import psycopg
import pytest
import tiktoken
from litellm import BaseModel, completion, embedding
from pypdf import PdfReader

from table_reclamation.facade.table_reclamation import AccessPlanner

# ==========================================
# 1. ENVIRONMENT & GLOBAL CONFIGURATION
# ==========================================
dotenv.load_dotenv()

# Dual-model setup for cost/compute efficiency
CHEAP_MODEL = "ollama/gemma4:e2b"     # Used for quick schema inference
EXPENSIVE_MODEL = "ollama/gemma4:31b"  # Used for heavy data extraction
MAX_TOKEN_SIZE = 16384

EMBEDDING_MODEL = "ollama/embeddinggemma:300m"
API_BASE = "http://host.docker.internal:11434"


# Global tracker for all extracted results
GLOBAL_RESULTS = []

# ==========================================
# 2. PYDANTIC SCHEMAS
# ==========================================


class Data(BaseModel):
    header: list[str]
    data: list[list[str]]
    explanation: str


class DataList(BaseModel):
    data: list[Data]


class SchemaInference(BaseModel):
    inferred_headers: list[str]

# ==========================================
# 3. HELPER FUNCTIONS
# ==========================================


def print_tabular_data(title: str, headers: list[str], data: list[list[str]]):
    """Dynamically sizes and prints data in a clean ASCII table."""
    print(f"\n{'='*10} {title} {'='*10}")
    if not data:
        print("No data extracted.\n")
        return

    headers = headers or [f"Col {i+1}" for i in range(len(data[0]))]

    col_widths = [len(str(h)) for h in headers]
    for row in data:
        for i, cell in enumerate(row):
            if i < len(col_widths):
                col_widths[i] = max(col_widths[i], len(str(cell)))
            else:
                col_widths.append(len(str(cell)))

    row_format = " | ".join([f"{{:<{w}}}" for w in col_widths])
    separator = "-" * (sum(col_widths) + 3 * len(col_widths) - 1)

    print(separator)
    print(row_format.format(*headers))
    print(separator)

    for row in data:
        padded_row = [str(item) for item in row] + \
            [""] * (len(headers) - len(row))
        print(row_format.format(*padded_row[:len(headers)]))

    print(separator + "\n")


def chunk_text(text: str, max_tokens: int, model: str = "gpt-4") -> list[str]:
    """Splits text into token-limited chunks for LLM processing."""
    enc = tiktoken.encoding_for_model(model)
    lines = text.split("\n")

    chunks = []
    current_chunk = []
    current_tokens = 0

    for line in lines:
        if not line.strip():
            continue
        row = line + "\n"
        tokens = len(enc.encode(row))
        if current_tokens + tokens > max_tokens:
            if current_chunk:
                chunks.append("".join(current_chunk))
            current_chunk = [row]
            current_tokens = tokens
        else:
            current_chunk.append(row)
            current_tokens += tokens

    if current_chunk:
        chunks.append("".join(current_chunk))

    return chunks


def extract_pdf_text(file_path: Path) -> str:
    """Safely extracts and concatenates all text from a PDF."""
    if not file_path.exists():
        print(f"Error: File not found at {file_path}")
        return ""
    try:
        reader = PdfReader(file_path)
        return "".join(page.extract_text() or "" for page in reader.pages)
    except Exception as e:
        print(f"Error reading {file_path}: {e}")
        return ""


# ==========================================
# 4. MAIN TEST LOGIC
# ==========================================
QUESTIONS = [
    # "How can I solve a linear system 4x4?",
    # "Fetch the assessments of student number 1273 and student number 3409",
    # "What is Diagonalization?",
    # "Give me a worked examples of the Product Rules",
    "Who is Eminem?"
]


@pytest.mark.parametrize("question", QUESTIONS)
def test_generate_prompt(planner_mathe_split: AccessPlanner, question: str):
    planner_mathe_split.generate_stats()
    plan = planner_mathe_split.generate_plan(question)
    print(f"\nPlan generated: {plan}")

    docs_to_check = []
    query_type = "RAG"
    base_dir = Path(
        "/workspaces/Table-Reclamation-Demo/data/mathe_unstructured_dataset")

    if plan:
        print("EXECUTION - TABLE RECLAMATION")
        pass

    # ---------------------------------------------------------
    # PHASE 1: Determine Documents to Check
    # ---------------------------------------------------------
    elif not plan:
        # Bi-encoder Route (RAG) - Get top 50 to allow deep pagination
        print("EXECUTION - RAG")
        conn_info = "dbname=rag user=postgres password=password host=db_rag port=5432"
        with psycopg.connect(conn_info) as conn:
            with conn.cursor() as cur:
                response = embedding(model=EMBEDDING_MODEL, input=[
                                     question], api_base=API_BASE)
                query = """
                    SELECT name, embedding <-> %s::vector AS distance
                    FROM items
                    ORDER BY distance
                    LIMIT 50;
                """
                cur.execute(query, (response.data[0]["embedding"],))
                all_results = cur.fetchall()

                if not all_results:
                    print("No items found in database table.")
                    return

                docs_to_check = [row[0] for row in all_results]

        # ---------------------------------------------------------
        # PHASE 2: Fast Schema Inference (CHEAP LLM)
        # ---------------------------------------------------------
        print(
            f"\n➔ Inferring canonical headers using CHEAP LLM ({CHEAP_MODEL})...")
        try:
            schema_response = completion(
                model=CHEAP_MODEL,
                messages=[
                    {
                        "role": "system",
                        "content": (
                            "You are a database architect engine.\n"
                            "Analyze the user's question and extract ONLY the column/attribute headers "
                            "required to construct a target structured table answering their request.\n"
                            "Normalize headers to lowercase with no spaces (use underscores).\n"
                            "You MUST return valid JSON exactly matching this format: {\"inferred_headers\": [\"col1\", \"col2\"]}"
                        )
                    },
                    {"role": "user", "content": f"Question: {question}"}
                ],
                response_format={"type": "json_object",
                                 "schema": SchemaInference.model_json_schema()},
                api_base=API_BASE,
                temperature=0,
            )
            parsed_schema = json.loads(
                schema_response.choices[0].message.content)
            raw_headers = parsed_schema.get("inferred_headers") or []
            canonical_headers = [h.strip().lower() for h in raw_headers]
        except Exception as e:
            print(
                f"🔴 Schema inference failed: {e}. Falling back to empty tracking.")
            canonical_headers = []

        print(f"   Headers identified: {canonical_headers}")

        # ---------------------------------------------------------
        # PHASE 3: Iterative Extraction Loop (EXPENSIVE LLM)
        # ---------------------------------------------------------
        unified_data_rows = []
        total_walltime = 0.0
        doc_index = 0

        while doc_index < len(docs_to_check):
            # Grab the next batch of up to 3 documents
            batch = docs_to_check[doc_index: doc_index + 3]
            if not batch:
                break

            data_found_in_batch = False
            print(f"\n➔ Checking next batch of up to 3 documents...")

            for document in batch:
                doc_index += 1
                print(
                    f"[Doc {doc_index}/{len(docs_to_check)}] Analyzing '{document}' with EXPENSIVE LLM ({EXPENSIVE_MODEL})...")

                text = extract_pdf_text(base_dir / f"{document}.pdf")
                if not text:
                    continue

                chunks = chunk_text(text, max_tokens=MAX_TOKEN_SIZE)
                document_data, document_headers = [], None

                start_time = time.time()
                for chunk in chunks:
                    try:
                        response = completion(
                            model=EXPENSIVE_MODEL,
                            messages=[
                                {"role": "system", "content": f"""
                            You are an expert Data Extraction Engine.

                            Your task:
                            1. Analyze the USER QUESTION to identify the implicit table headers (the attributes or columns being asked for).
                            2. Map those headers positionally to the raw space-separated DATASET rows.
                            3. Extract and return the matching records in the strict JSON format below.

                            --------------------------------
                            STRICT OUTPUT RULES
                            --------------------------------
                            1. Output MUST be valid JSON only. No markdown wrappers (like ```json), and no text before or after.
                            2. Output MUST match EXACTLY this schema:
                            3. Do NOT modify the headers.

                            {{
                            "header": {json.dumps(canonical_headers)},
                            "data": [["value1", "value2", "..."]],
                            "explanation": "Brief explanation of how the criteria was met"
                            }}

                            3. The "header" array MUST represent the columns inferred from the question (e.g., ["country", "per_capita_sales", "consumption"]).
                            4. DO NOT wrap the entire JSON output in a list.
                            5. If NO rows in the text dataset match the criteria in the question → return:
                            "data": []
                            6. Each row inside the "data" array MUST have the EXACT same number of elements as your inferred "header" array.
                            7. NEVER return malformed or partial rows.

                            --------------------------------
                            DATASET PARSING RULES
                            --------------------------------
                            - The dataset input is RAW TEXT where rows are separated by newlines and values are separated by spaces.
                            - There are NO column headers inside the raw text dataset. You must look at the question to understand what each space-separated position represents.
                            - Line-by-line, parse the rows, evaluate if they answer the user's question, and map the values to your inferred headers.

                            IMPORTANT FINAL CHECK (before answering):
                            - Is JSON structurally valid? ✔
                            - Does each data row length perfectly match the output header length? ✔
                            - Did you exclude any conversational preambles or postscripts? ✔ (Only output raw JSON)
                            """},
                                {"role": "user", "content": f"""
                            DATASET:
                            {chunk}

                            SQL QUERY:
                            {question}

                            Return ONLY the JSON.
                            """
                                }],
                            response_format={"type": "json_object",
                                             "schema": Data.model_json_schema()},
                            api_base=API_BASE,
                            timeout=7200,
                        )
                        parsed = json.loads(
                            response.choices[0].message.content)
                        if document_headers is None:
                            document_headers = [h.strip().lower()
                                                for h in parsed.get("header", [])]
                        document_data.extend(parsed.get("data", []))
                    except Exception as e:
                        print(f"    Skipping invalid chunk in {document}: {e}")

                total_walltime += (time.time() - start_time)

                # Header Alignment & Verification
                valid_doc_rows = []
                if document_data:
                    for row in document_data:
                        # Drop rows that are entirely N/A or empty
                        if not all(str(cell).strip().upper() in ("N/A", "", "NONE") for cell in row):
                            row_dict = {h: (row[idx] if idx < len(row) else "N/A")
                                        for idx, h in enumerate(document_headers or [])}
                            aligned_row = [row_dict.get(col, "N/A")
                                           for col in canonical_headers]
                            valid_doc_rows.append(aligned_row)

                if valid_doc_rows:
                    print(f"  ✔️ Valid data found in '{document}'.")
                    unified_data_rows.extend(valid_doc_rows)
                    data_found_in_batch = True
                    break  # Short-circuit out of the current 3-document batch
                else:
                    print(f"  ❌ No relevant data in '{document}'.")

            # ---------------------------------------------------------
            # PHASE 4: User Prompt for Continuation
            # ---------------------------------------------------------
            if doc_index < len(docs_to_check):
                ans = input(
                    f"\nChecked {doc_index} documents so far. Do you want to check more? (y/n): ").strip().lower()
                if ans != 'y':
                    break

        # ---------------------------------------------------------
        # PHASE 5: Deduplication & Output
        # ---------------------------------------------------------
        if not unified_data_rows:
            print("\nNo data found after checking selected documents.")
            return

        # Exact Deduplication
        unique_data = list({tuple(row) for row in unified_data_rows})

        final_aggregated_result = {
            "query_type": query_type,
            "question": question,
            "headers": canonical_headers,
            "data": unique_data,
            "total_rows": len(unique_data),
            "total_walltime_seconds": round(total_walltime, 2)
        }

        GLOBAL_RESULTS.append(final_aggregated_result)
        print_tabular_data(
            f"Results for: {question[:30]}...", canonical_headers, unique_data)
        print(GLOBAL_RESULTS)
