import json
import math
import os
import re
import time
from functools import partial
from multiprocessing import Pool, cpu_count
from os import read
from pathlib import Path

import dotenv
import matplotlib.pyplot as plt
import psycopg
import pytest
import tiktoken
from litellm import BaseModel, completion, embedding
from matplotlib.ticker import ScalarFormatter
from pgvector.psycopg import register_vector
from pypdf import PdfReader

# --- adjustText IMPORT WITH GRACEFUL FALLBACK ---
try:
    from adjustText import adjust_text
    HAS_ADJUST_TEXT = True
except ImportError:
    print("\n[INFO] 'adjustText' library not found. Labels in the plot may overlap.")
    print("[INFO] To fix this, run: pip install adjustText\n")
    HAS_ADJUST_TEXT = False
# ------------------------------------------------

from table_reclamation.facade.table_reclamation import AccessPlanner

dotenv.load_dotenv()
# MODEL = "ollama/qwen3.6:35b" #Don't forget to toggle off structured output.
MODEL = "ollama/gemma4:31b"
# MODEL = "ollama/qwen3.5:122b"

GPU = "H200"

MODELNAME = MODEL[7:]


################# For Tiktoken tokenizer #################


def chunk_with_header(text, max_tokens, model="gpt-4"):
    enc = tiktoken.encoding_for_model(model)

    lines = text.split("\n")

    header = None
    chunks = []
    current_chunk = []
    current_tokens = 0
    started = False  # controls when chunking begins

    for line in lines:
        if not line.strip():
            continue

        # Detect header
        if not started:
            if "student_id" in line:  # replace with your detection logic if needed
                header = line
                started = True
            continue  # ignore everything before header

        # From here: only real table rows
        row = line + "\n"
        tokens = len(enc.encode(row))

        if current_tokens + tokens > max_tokens:
            if current_chunk:
                chunk_text = header + "\n" + "".join(current_chunk)
                chunks.append(chunk_text)

            current_chunk = [row]
            current_tokens = tokens
        else:
            current_chunk.append(row)
            current_tokens += tokens

    # Final chunk
    if current_chunk:
        chunk_text = header + "\n" + "".join(current_chunk)
        chunks.append(chunk_text)

    return chunks


def chunk_without_header(text, max_tokens, model="gpt-4"):
    enc = tiktoken.encoding_for_model(model)

    lines = text.split("\n")

    # header = None
    chunks = []
    current_chunk = []
    current_tokens = 0
    # started = False  # controls when chunking begins

    for line in lines:
        if not line.strip():
            continue

        row = line + "\n"
        tokens = len(enc.encode(row))

        if current_tokens + tokens > max_tokens:
            if current_chunk:
                chunks.append(current_chunk)

            current_chunk = [row]
            current_tokens = tokens
        else:
            current_chunk.append(row)
            current_tokens += tokens

    if current_chunk:
        chunks.append(current_chunk)

    return chunks


############## PDF text extract in parallel #############


def extract_page_text(i, file_path):
    try:
        reader = PdfReader(file_path)  # reopen inside process
        return reader.pages[i].extract_text()
    except Exception as e:
        print(f"Error processing page {i}: {e}")
        return ""  # return empty string on error to avoid breaking the whole process


def filter_by_dynamic_zscore(results: list[tuple[str, float]], k: float = 2.0):
    """Filters a sorted list of (name, distance) tuples by evaluating

    the statistical outliers among consecutive distance gaps.
    """
    if len(results) <= 2:
        return results  # Not enough data points to compute standard deviation safely

    distances = [row[1] for row in results]

    # 1. Calculate the progression gap between each sequential document
    gaps = [distances[i] - distances[i - 1] for i in range(1, len(distances))]

    # 2. Calculate the mean ($\mu$) of the gaps
    mean_gap = sum(gaps) / len(gaps)

    # 3. Calculate the standard deviation ($\sigma$) of the gaps
    variance = sum((g - mean_gap) ** 2 for g in gaps) / len(gaps)
    std_gap = math.sqrt(variance)

    # 4. Set the dynamic maximum gap threshold ($\mu + k \cdot \sigma$)
    # Adjusting $k$ makes the algorithm more strict (lower $k$) or lenient (higher $k$)
    dynamic_max_gap = mean_gap + (k * std_gap)

    filtered = [results[0]]

    # 5. Loop through and halt when a gap exceeds the dynamic threshold
    for i in range(1, len(results)):
        gap = distances[i] - distances[i - 1]

        if gap > dynamic_max_gap:
            print(f"\n Dynamic Cutoff Triggered at Rank {i+1}!")
            print(
                f"   Gap between '{results[i-1][0]}' and '{results[i][0]}' was {gap:.4f}."
            )
            print(
                f"   Dynamic Maximum Allowed Gap was {dynamic_max_gap:.4f}.\n")
            break

        filtered.append(results[i])

    return filtered


QUESTIONS = [
    "How can I solve a linear system 4x4?",
    "What are the basic concepts of learning linear optimization?",
    "How do I begin my study about linear optimization?",
    "What is the role of the objective function in finding the optimum solution in a linear optimization problem?",
    "What is the role of constraints in a linear optimization problem?",
    "How is the optimal solution usually found in a linear optimization problem?",
    "Fetch the assessments of student number 1273 and student number 3409",  # 2 SQL Sequences
]

QUESTIONS_RAG = [
    "What is Product Rule?",
    "What is Diagonalization?"
]


@pytest.mark.parametrize("question", QUESTIONS)
def test_generate_mathe_plan(planner_mathe_split: AccessPlanner, question: str):
    planner_mathe_split.generate_stats()
    plan = planner_mathe_split.generate_plan(question)
    print(plan)
    assert len(plan) > 1


@pytest.mark.parametrize("question", QUESTIONS)
def test_generate_prompt(planner_mathe_split: AccessPlanner, question: str):
    planner_mathe_split.generate_stats()
    plan = planner_mathe_split.generate_plan(question)
    print(plan)

    class Data(BaseModel):
        header: list[str]
        data: list[list[str]]
        explanation: str

    class DataList(BaseModel):
        data: list[Data]

    for p in plan:
        p.type = 'Document'
        # p.sql = "SELECT id, student_id FROM mathe.assessment_10 WHERE student_id IN ('3409')"

        if p.type == 'Query':
            pass
        elif p.type == 'Document':
            # 1. Use regex to find the word immediately following 'FROM'
            # [^\s]+ matches any character that isn't a space (the table name)
            match = re.search(r"FROM\s+([^\s]+)", p.sql, re.IGNORECASE)
            print(match)
            if match:
                table_name = match.group(1)

                # 2. Dynamically construct the file path using f-strings
                base_dir = Path(
                    "/workspaces/Table-Reclamation-Demo/data/mathe_splitted")
                file_path = base_dir / f"{table_name}.pdf"

                # 3. Process the file if it exists
                if file_path.exists():
                    DocumentReader = PdfReader(file_path)
                    text = ""
                    for i in range(len(DocumentReader.pages)):
                        text += DocumentReader.pages[i].extract_text()
                else:
                    print(f"Error: File not found at {file_path}")
            else:
                print(
                    "Error: Could not find a table name after 'FROM' in the SQL query.")

            for i in range(100):  # Repetitive execution for Whisker plot
                # more precise check around 1K~4K tokens
                context_sizes = [
                    2**i for i in range(10, 18)] + [512, 1536, 2560, 3072, 3584, 4608]
                context_sizes.sort()
                results_log = []

                for context_size in context_sizes:
                    print(f"\n=== Testing context size: {context_size} ===")

                    chunks = chunk_without_header(
                        text, max_tokens=context_size)
                    all_results = []

                    start_time = time.time()
                    for i, chunk in enumerate(chunks):
                        print(f"Processing chunk {i}/{len(chunks)}")

                        response = completion(
                            model=MODEL,
                            messages=[
                                {"role": "system", "content": """
                                    You are a deterministic SQL execution engine.

                                    Your task:
                                    Execute the SQL query EXACTLY on the provided dataset.

                                    --------------------------------
                                    STRICT RULES (MANDATORY)
                                    --------------------------------

                                    1. Output MUST be valid JSON only. No text before or after.
                                    2. Output MUST match EXACTLY this schema:

                                    {
                                    "header": ["column1", "..."],
                                    "data": [["value1", "..."]],
                                    "explanation": "short explanation"
                                    }

                                    3. DO NOT wrap output in a list.
                                    4. DO NOT return multiple JSON objects.
                                    5. DO NOT repeat the query.
                                    6. DO NOT hallucinate values.
                                    7. ONLY return rows that EXACTLY match the WHERE condition.
                                    8. If NO rows match → return:
                                    "data": []

                                    9. Each row in "data" MUST have the SAME number of columns as "header".
                                    10. NEVER return malformed rows (e.g., ["123"] if 2 columns expcted).

                                    --------------------------------
                                    DATASET RULES
                                    --------------------------------

                                    - Dataset is RAW TEXT (space-separated)
                                    - First line = column names
                                    - Each next line = one row
                                    - You MUST manually parse rows
                                    - Columns are separated by spaces

                                    --------------------------------
                                    SQL RULES
                                    --------------------------------

                                    - Only use the provided dataset
                                    - Apply WHERE conditions strictly
                                    - SELECT only requested columns
                                    - DISTINCT = remove duplicates

                                    IMPORTANT FINAL CHECK (before answering):
                                    - Is JSON valid? ✔
                                    - Does each row match header length? ✔
                                    - Any hallucinated values? ✘
                                    - Any partial matches? ✘

                                    If any rule is violated → FIX before returning.
                                    """},
                                {"role": "user", "content": f"""
                                    DATASET:
                                    {chunk}

                                    SQL QUERY:
                                    {p.sql}

                                    Return ONLY the JSON.
                                    """
                                 }],
                            response_format=Data,
                            api_base="http://host.docker.internal:11434",
                            timeout=7200,
                            # stream=False,
                            # extra_body={
                            #     "options": {
                            #         "multi_token_prediction": False,
                            #         "temperature": 0,
                            #         # "num_predict": 512
                            #     }
                            # }
                        )

                        print(response)
                        all_results.append(response)

                    walltime = time.time() - start_time
                    print(all_results)
                    print("Received={}".format(all_results))

                    all_data = []
                    headers = None

                    for res in all_results:
                        try:
                            content = res.choices[0].message.content
                            parsed = json.loads(content)

                            if headers is None:
                                headers = parsed.get("header", [])

                            all_data.extend(parsed.get("data", []))

                        except Exception as e:
                            print("Skipping invalid response:", e)

                    result = {
                        "header": headers,
                        "data": all_data
                    }

                    # ---------------------------------------------------------
                    # CONFUSION MATRIX & METRICS LOGIC
                    # ---------------------------------------------------------

                    data_lines = [line for line in text.split(
                        "\n") if line.strip() and "student_id" not in line]
                    total_dataset_rows = len(data_lines)

                    actual_p = 100  # Known expected positives for student_id '3409'
                    actual_n = max(0, total_dataset_rows - actual_p)

                    tp = 0
                    fp = 0

                    for d in result["data"]:
                        # Check if row has at least 2 elements, first element is in valid_ids, and second is 3409
                        if len(d) >= 2 and str(d[1]) == '3409':
                            tp += 1
                        else:
                            fp += 1

                    fn = max(0, actual_p - tp)
                    tn = max(0, actual_n - fp)

                    # Advanced Metrics Calculations (with zero-division protection)
                    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
                    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
                    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
                    fall_out = fp / (fp + tn) if (fp + tn) > 0 else 0.0

                    f1_denominator = precision + recall
                    f1_score = 2 * (precision * recall) / \
                        f1_denominator if f1_denominator > 0 else 0.0

                    total_population = tp + tn + fp + fn
                    accuracy = (tp + tn) / \
                        total_population if total_population > 0 else 0.0

                    results_log.append({
                        "GPU": GPU,
                        "context_size": context_size,
                        "walltime": walltime,
                        "TP": tp,
                        "TN": tn,
                        "FP": fp,
                        "FN": fn,
                        "Recall": recall,
                        "Precision": precision,
                        "Specificity": specificity,
                        "Fall-out": fall_out,
                        "F1-Score": f1_score,
                        "Accuracy": accuracy,
                        "num_chunks": len(chunks),
                        "num_rows": len(result["data"]),
                        "result": result
                    })

                # --- JSON EXPORT CODE ---
                log_file_path = "logs.json"
                existing_logs = {}

                if os.path.exists(log_file_path) and os.path.getsize(log_file_path) > 0:
                    try:
                        with open(log_file_path, "r", encoding="utf-8") as f:
                            existing_logs = json.load(f)
                    except json.JSONDecodeError:
                        print(
                            f"Warning: '{log_file_path}' contains invalid JSON. Starting fresh.")

                if MODELNAME in existing_logs:
                    existing_logs[MODELNAME].extend(results_log)
                else:
                    existing_logs[MODELNAME] = results_log

                with open(log_file_path, "w", encoding="utf-8") as f:
                    json.dump(existing_logs, f, indent=4)

            # --- PRECISION-RECALL PLOT CODE ---
            import matplotlib.pyplot as plt

            recall_list = [r["Recall"] for r in results_log]
            precision_list = [r["Precision"] for r in results_log]
            sizes = [r["context_size"] for r in results_log]

            fig, ax = plt.subplots(figsize=(10, 8))

            # 1. Plot the main trajectory line and points
            ax.plot(
                recall_list,
                precision_list,
                marker='o',
                markersize=8,
                color='tab:purple',
                linestyle='-',
                label='Model Performance Trajectory'
            )

            # 2. Collect all text annotation objects
            texts = []
            for i, txt in enumerate(sizes):
                texts.append(ax.annotate(
                    f"{txt}",
                    (recall_list[i], precision_list[i]),
                    fontsize=9,
                    bbox=dict(boxstyle="round,pad=0.3",
                              fc="white", ec="gray", alpha=0.8)
                ))

            # 3. Perform the automatic label adjustment
            if HAS_ADJUST_TEXT:
                print(
                    "[INFO] Adjusting labels to prevent overlap... this may take a moment.")
                adjust_text(
                    texts,
                    x=recall_list,
                    y=precision_list,
                    expand_points=(2.0, 2.0),
                    force_text=(0.3, 0.6),
                    force_points=(0.2, 0.5),
                    lim=500,
                    arrowprops=dict(arrowstyle="->",
                                    color='gray', lw=0.5, alpha=0.7)
                )
            else:
                # Fallback if adjustText is not installed
                for t in texts:
                    t.set_va('center')
                    t.set_ha('center')

            ax.set_title("Precision-Recall Space by Context Size")
            ax.set_xlabel("Recall (True Positive Rate)")
            ax.set_ylabel("Precision (Positive Predictive Value)")

            # Padding
            ax.set_xlim([-0.08, 1.08])
            ax.set_ylim([-0.08, 1.08])

            ax.grid(True, which="both", ls="--", alpha=0.5)
            ax.legend(loc='upper right')

            fig.tight_layout()

            # Save and show
            plt.savefig(f"pr_plot_{MODELNAME}.png")
            plt.show()
            plt.close()


def test_litellm():
    response = completion(
        model=MODEL,
        messages=[
            {"content": "respond in 20 words. who are you?", "role": "user"}],
        api_base="http://host.docker.internal:11434"
    )
    print(response)


@pytest.mark.parametrize("question", QUESTIONS_RAG)
def test_embedding(question: str):
    user_input = question
    conn_info = "dbname=rag user=postgres password=password host=db_rag port=5432"

    primary_context_documents = []

    # 0. Define Schemas at Top-Level (Outside the test function)
    class Data(BaseModel):
        header: list[str]
        data: list[list[str]]
        explanation: str

    class DataList(BaseModel):
        data: list[Data]

    # [1. DB Retrieval Block remains identical]
    with psycopg.connect(conn_info) as conn:
        with conn.cursor() as cur:
            response = embedding(
                model="ollama/embeddinggemma:300m",
                input=[user_input],
                api_base="http://host.docker.internal:11434",
            )
            embeddings = response.data[0]["embedding"]
            query = """
                SELECT name, embedding <-> %s::vector AS distance
                FROM items
                ORDER BY distance;
            """
            cur.execute(query, (embeddings,))
            all_results = cur.fetchall()

            if not all_results:
                pytest.fail(
                    "No items found in database table to run RAG evaluation.")

            filtered_results = filter_by_dynamic_zscore(all_results, k=2.0)
            primary_context_documents = [row[0] for row in filtered_results]

    # --- NEW: STATIC SCHEMA PARSING FROM USER QUERY ---
    print(
        f"\n➔ Inferring canonical headers directly from query: '{user_input}'")

    # We define a minimal Pydantic model just to safely capture the standalone schema
    class SchemaInference(BaseModel):
        inferred_headers: list[str]

    try:
        schema_response = completion(
            model=MODEL,
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
                {"role": "user", "content": f"Question: {user_input}"}
            ],
            response_format={"type": "json_object",
                             "schema": SchemaInference.model_json_schema()},
            api_base="http://host.docker.internal:11434",
            temperature=0,  # Strict determinism
        )
        parsed_schema = json.loads(schema_response.choices[0].message.content)

        # Fallback chain: Check for 'inferred_headers', then 'columns', then 'header'
        raw_headers = (
            parsed_schema.get("inferred_headers") or
            parsed_schema.get("columns") or
            parsed_schema.get("header") or
            []
        )

        canonical_headers = [h.strip().lower() for h in raw_headers]
        print(f"➔ Definitive Canonical Headers Set: {canonical_headers}\n")

    except Exception as e:
        print(
            f"🔴 Schema inference failed: {e}. Falling back to default empty tracking.")
        canonical_headers = []

    # --- TRACKERS FOR LOGGING ---
    all_document_evaluation_logs = []
    unified_data_rows = []
    total_walltime = 0.0

    # 4. Processing Loop
    for document in primary_context_documents:
        base_dir = Path(
            "/workspaces/Table-Reclamation-Demo/data/mathe_unstructured_dataset")
        file_path = base_dir / f"{document}.pdf"
        text = ""

        if file_path.exists():
            DocumentReader = PdfReader(file_path)
            for i in range(len(DocumentReader.pages)):
                text += DocumentReader.pages[i].extract_text() or ""
        else:
            continue

        context_sizes = [16384]
        for context_size in context_sizes:
            chunks = chunk_without_header(text, max_tokens=context_size)
            results_from_all_chunks = []

            start_time = time.time()
            for i, chunk in enumerate(chunks):
                response = completion(
                    model=MODEL,
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
                            {user_input}

                            Return ONLY the JSON.
                            """
                         }],
                    response_format={"type": "json_object",
                                     "schema": Data.model_json_schema()},
                    api_base="http://host.docker.internal:11434",
                    timeout=7200,
                )
                results_from_all_chunks.append(response)

            walltime = time.time() - start_time
            total_walltime += walltime

            document_data = []
            document_headers = None

            for res in results_from_all_chunks:
                try:
                    content = res.choices[0].message.content
                    parsed = json.loads(content)

                    if document_headers is None:
                        document_headers = [h.strip().lower()
                                            for h in parsed.get("header", [])]

                    document_data.extend(parsed.get("data", []))
                except Exception as e:
                    print(f"Skipping invalid response in {document}:", e)

            # --- HEADER ALIGNMENT ENGINE (Now maps against your query schema) ---
            if document_data:
                if document_headers == canonical_headers:
                    unified_data_rows.extend(document_data)
                else:
                    # Defensive mapping: aligning variant extractions back to the query baseline schema
                    for row in document_data:
                        aligned_row = []
                        row_dict = {
                            h: (row[idx] if idx < len(row) else "N/A")
                            for idx, h in enumerate(document_headers or [])
                        }
                        for target_col in canonical_headers:
                            aligned_row.append(row_dict.get(target_col, "N/A"))
                        unified_data_rows.append(aligned_row)

            document_log = {
                "document": document,
                "context_size": context_size,
                "extracted_headers": document_headers,
                "extracted_data": document_data,
                "walltime_seconds": round(walltime, 2)
            }
            all_document_evaluation_logs.append(document_log)

    # --- 5. SYNTHESIZE LOGS ---

    # 1. Filter out rows where every single cell is exactly "N/A"
    cleaned_data_rows = [
        row for row in unified_data_rows
        if not all(str(cell).strip().upper() == "N/A" for cell in row)
    ]

    # 2. Programmatic Exact Deduplication (Reduces token load for the LLM)
    unique_data_rows = []
    seen = set()
    for row in cleaned_data_rows:
        row_tuple = tuple(row)
        if row_tuple not in seen:
            seen.add(row_tuple)
            unique_data_rows.append(row)

    # --- 3. LLM SEMANTIC DEDUPLICATION ---
    print("\n➔ Running LLM Semantic Deduplication...")
    semantically_unique_rows = unique_data_rows  # Default fallback

    if unique_data_rows:
        try:
            dedup_response = completion(
                model=MODEL,
                messages=[
                    {
                        "role": "system",
                        "content": (
                            "You are an expert Data Cleaner.\n"
                            "Your task is to review the provided dataset and remove semantically duplicate rows.\n"
                            "Many rows contain the exact same information but have slight variations in:\n"
                            "- Punctuation (e.g., trailing periods)\n"
                            "- Phrasing (e.g., 'The Product Rule is used' vs 'Product Rule is used')\n"
                            "- Mathematical notation or syntax.\n"
                            "When you find semantic duplicates, keep the most comprehensive and grammatically correct version, and discard the rest.\n"
                            "Do NOT alter the headers. Remove empty or nonsensical rows (like a single 'e').\n"
                            "You MUST return valid JSON exactly matching this structure:\n"
                            "{\n"
                            f'  "header": {json.dumps(canonical_headers)},\n'
                            '  "data": [["val1", "val2"]],\n'
                            '  "explanation": "Brief explanation of what was removed"\n'
                            "}"
                        )
                    },
                    {
                        "role": "user",
                        "content": f"DATASET TO CLEAN:\n{json.dumps(unique_data_rows, indent=2)}\n\nReturn ONLY raw JSON."
                    }
                ],
                # Reusing your top-level Data schema to enforce structural integrity
                response_format={"type": "json_object",
                                 "schema": Data.model_json_schema()},
                api_base="http://host.docker.internal:11434",
                temperature=0,  # Must be 0 for deterministic data cleaning
            )

            parsed_dedup = json.loads(
                dedup_response.choices[0].message.content)

            # Ensure the LLM didn't hallucinate a different structure
            if "data" in parsed_dedup:
                semantically_unique_rows = parsed_dedup["data"]
                removed_count = len(unique_data_rows) - \
                    len(semantically_unique_rows)
                print(
                    f"➔ Semantic deduplication successful. Removed {removed_count} near-duplicate rows.")
                print(
                    f"➔ Cleaning explanation: {parsed_dedup.get('explanation', 'None provided')}")

        except Exception as e:
            print(
                f"🔴 LLM Deduplication failed: {e}. Falling back to exact match programmatic deduplication.")

    # 4. Build the final payload
    final_aggregated_result = {
        "header": canonical_headers,
        "data": semantically_unique_rows,
        "total_rows": len(semantically_unique_rows),
        "total_walltime_seconds": round(total_walltime, 2)
    }

    print("\n=============================================")
    print("      FINAL UNIFIED RUN RESULTS (ALL DOCS)")
    print("=============================================")
    print(json.dumps(final_aggregated_result, indent=2))

    log_file_path = Path(
        "/workspaces/Table-Reclamation-Demo/tests/run_results.json")
    with open(log_file_path, "w") as f:
        json.dump({
            "test_question": user_input,
            "timestamp": time.time(),
            "aggregated_output": final_aggregated_result,
            "per_document_trace": all_document_evaluation_logs
        }, f, indent=4)

    assert len(final_aggregated_result["data"]) >= 0


@pytest.mark.parametrize("question", QUESTIONS)
def test_generate_prompt_case2_prejoin(planner_mathe_split: AccessPlanner, question: str):
    planner_mathe_split.generate_stats()
    plan = planner_mathe_split.generate_plan(question)
    print(plan)

    class Data(BaseModel):
        header: list[str]
        data: list[list[str]]
        explanation: str

    class DataList(BaseModel):
        data: list[Data]

    text = ""  # Initialize text list with empty strings
    sqls = [""] * len(plan)  # Initialize sqls list with empty strings
    results_log = []         # Move outside the loop to collect records cleanly

    for j, p in enumerate(plan):
        p.type = 'Document'

        ###################### hallucination check: LLM might generate a fake id ######################
        if j == 0:
            p.sql = "SELECT id, student_id FROM mathe.assessment_10 WHERE student_id IN ('3409')"
        elif j == 1:
            p.sql = "SELECT id, student_id FROM mathe.assessment_4 WHERE student_id IN ('1273')"
        ###############################################################################################

        sqls[j] = p.sql  # Store SQL for later use in metrics log

        if p.type == 'Query':
            pass
        elif p.type == 'Document':
            match = re.search(r"FROM\s+([^\s]+)", p.sql, re.IGNORECASE)
            print(match)
            if match:
                table_name = match.group(1)
                base_dir = Path(
                    "/workspaces/Table-Reclamation-Demo/data/mathe_splitted")
                file_path = base_dir / f"{table_name}.pdf"

                if file_path.exists():
                    DocumentReader = PdfReader(file_path)
                    for i in range(len(DocumentReader.pages)):
                        text += DocumentReader.pages[i].extract_text()
                else:
                    print(f"Error: File not found at {file_path}")
            else:
                print(
                    "Error: Could not find a table name in the SQL query.")

    for run_idx in range(100):
        # context_sizes = [
        #     2**idx for idx in range(10, 18)] + [512, 1536, 2560, 3072, 3584, 4608]
        context_sizes = [32768]
        context_sizes.sort()

        for context_size in context_sizes:
            print(f"\n=== Testing context size: {context_size} ===")
            chunks = chunk_with_header(text, max_tokens=context_size)
            all_results = []

            start_time = time.time()
            for chunk_idx, chunk in enumerate(chunks):
                print(f"Processing chunk {chunk_idx}/{len(chunks)}")

                response = completion(
                    model=MODEL,
                    messages=[
                        {"role": "system", "content": """
                            You are a deterministic SQL execution engine.

                            Your task:
                            Execute the list of SQL queries EXACTLY on the provided dataset.

                            --------------------------------
                            STRICT RULES (MANDATORY)
                            --------------------------------

                            1. Output MUST be valid JSON only. No text before or after.
                            2. Output MUST match EXACTLY this schema:

                            {
                            "header": ["column1", "..."],
                            "data": [["value1", "..."]],
                            "explanation": "short explanation"
                            }

                            3. DO NOT wrap output in a list.
                            4. DO NOT return multiple JSON objects.
                            5. DO NOT repeat the query.
                            6. DO NOT hallucinate values.
                            7. ONLY return rows that EXACTLY match the WHERE condition.
                            8. If NO rows match → return:
                            "data": []

                            9. Each row in "data" MUST have the SAME number of columns as "header".
                            10. NEVER return malformed rows (e.g., ["123"] if 2 columns expcted).

                            --------------------------------
                            DATASET RULES
                            --------------------------------

                            - Dataset is RAW TEXT (space-separated)
                            - First line = column names
                            - Each next line = one row
                            - You MUST manually parse rows
                            - Columns are separated by spaces

                            --------------------------------
                            SQL RULES
                            --------------------------------

                            - Only use the provided dataset
                            - Apply WHERE conditions strictly
                            - SELECT only requested columns
                            - DISTINCT = remove duplicates
                            
                            IMPORTANT FINAL CHECK (before answering):
                            - Is JSON valid? ✔
                            - Does each row match header length? ✔
                            - Any hallucinated values? ✘
                            - Any partial matches? ✘

                            If any rule is violated → FIX before returning.
                            """},
                        {"role": "user", "content": f"DATASET:{chunk} SQL QUERY: {sqls} Return ONLY the JSON."}
                    ],
                    response_format=Data,
                    api_base="http://host.docker.internal:11434",
                    timeout=7200,
                )
                print(response)
                all_results.append(response)

            walltime = time.time() - start_time
            all_data = []
            headers = None

            for res in all_results:
                try:
                    content = res.choices[0].message.content
                    parsed = json.loads(content)
                    if headers is None:
                        headers = parsed.get("header", [])
                    all_data.extend(parsed.get("data", []))
                except Exception as e:
                    print("Skipping invalid response:", e)

            result = {
                "header": headers,
                "data": all_data
            }

            match = re.search(
                r"SELECT\s+([a-zA-Z0-9_\*]+)", sqls[0], re.IGNORECASE)
            data_lines = [line for line in text.split(
                "\n") if line.strip() and match.group(1) not in line]
            total_dataset_rows = len(data_lines)

            actual_p = 200
            actual_n = max(0, total_dataset_rows - actual_p)

            tp = 0
            fp = 0

            for d in result["data"]:
                if len(d) >= 2 and (str(d[1]) == '3409' or str(d[1]) == '1273'):
                    tp += 1
                else:
                    fp += 1

            fn = max(0, actual_p - tp)
            tn = max(0, actual_n - fp)

            recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
            fall_out = fp / (fp + tn) if (fp + tn) > 0 else 0.0

            f1_denominator = precision + recall
            f1_score = 2 * (precision * recall) / \
                f1_denominator if f1_denominator > 0 else 0.0

            total_population = tp + tn + fp + fn
            accuracy = (tp + tn) / \
                total_population if total_population > 0 else 0.0

            results_log.append({
                "GPU": GPU,
                "context_size": context_size,
                "walltime": walltime,
                "TP": tp,
                "TN": tn,
                "FP": fp,
                "FN": fn,
                "Recall": recall,
                "Precision": precision,
                "Specificity": specificity,
                "Fall-out": fall_out,
                "F1-Score": f1_score,
                "Accuracy": accuracy,
                "num_chunks": len(chunks),
                "num_rows": len(result["data"]),
                "result": result
            })

            # --- JSON EXPORT CODE ---
            log_file_path = "logs_prejoined_queries.json"
            existing_logs = {}

            if os.path.exists(log_file_path) and os.path.getsize(log_file_path) > 0:
                try:
                    with open(log_file_path, "r", encoding="utf-8") as f:
                        existing_logs = json.load(f)
                except json.JSONDecodeError:
                    print(
                        f"Warning: '{log_file_path}' contains invalid JSON. Starting fresh.")

            if MODELNAME in existing_logs:
                existing_logs[MODELNAME].extend(results_log)
            else:
                existing_logs[MODELNAME] = results_log

            with open(log_file_path, "w", encoding="utf-8") as f:
                json.dump(existing_logs, f, indent=4)

    # --- PLOT GENERATION OUTSIDE OF EXECUTION LOOP ---
    if results_log:
        recall_list = [r["Recall"] for r in results_log]
        precision_list = [r["Precision"] for r in results_log]
        sizes = [r["context_size"] for r in results_log]

        fig, ax = plt.subplots(figsize=(10, 8))
        ax.plot(
            recall_list,
            precision_list,
            marker='o',
            markersize=8,
            color='tab:purple',
            linestyle='-',
            label='Model Performance Trajectory'
        )

        texts = []
        for idx, txt in enumerate(sizes):
            texts.append(ax.annotate(
                f"{txt}",
                (recall_list[idx], precision_list[idx]),
                fontsize=9,
                bbox=dict(boxstyle="round,pad=0.3",
                          fc="white", ec="gray", alpha=0.8)
            ))

        if HAS_ADJUST_TEXT:
            print("[INFO] Adjusting labels to prevent overlap...")
            adjust_text(
                texts,
                x=recall_list,
                y=precision_list,
                expand_points=(2.0, 2.0),
                force_text=(0.3, 0.6),
                force_points=(0.2, 0.5),
                lim=500,
                arrowprops=dict(arrowstyle="->", color='gray',
                                lw=0.5, alpha=0.7)
            )
        else:
            for t in texts:
                t.set_va('center')
                t.set_ha('center')

        ax.set_title("Precision-Recall Space by Context Size")
        ax.set_xlabel("Recall (True Positive Rate)")
        ax.set_ylabel("Precision (Positive Predictive Value)")
        ax.set_xlim([-0.08, 1.08])
        ax.set_ylim([-0.08, 1.08])
        ax.grid(True, which="both", ls="--", alpha=0.5)
        ax.legend(loc='upper right')

        fig.tight_layout()
        plt.savefig(f"pr_plot_{MODELNAME}.png")
        plt.show()
        plt.close()


@pytest.mark.parametrize("question", QUESTIONS)
def test_generate_prompt_case1_postjoin(planner_mathe_split: AccessPlanner, question: str):
    planner_mathe_split.generate_stats()
    plan = planner_mathe_split.generate_plan(question)
    print(plan)

    class Data(BaseModel):
        header: list[str]
        data: list[list[str]]
        explanation: str

    class DataList(BaseModel):
        data: list[Data]

    total_dataset_rows = 0

    for run_idx in range(100):
        texts = [""] * len(plan)  # Initialize text list with empty strings
        results_log = []         # Move outside the loop to collect records cleanly

        # + [512, 1536, 2560, 3072, 3584, 4608]
        context_sizes = [2**idx for idx in range(13, 18)]
        # context_sizes = [50000, 60000]
        context_sizes.sort()

        for context_size in context_sizes:
            print(f"\n=== Testing context size: {context_size} ===")
            # Initialize chunks list for this context size
            chunks = [""] * len(plan)
            all_results = []

            walltime = 0
            # To track time taken for each query/document
            elapsed_time = [0] * len(plan)

            for j, p in enumerate(plan):
                print(f"Processing query {j}/{len(plan)-1}")
                p.type = 'Document'

                ###################### hallucination check: LLM might generate a fake id ######################
                if j == 0:
                    p.sql = "SELECT id, student_id FROM mathe.assessment_10 WHERE student_id IN ('3409')"
                elif j == 1:
                    p.sql = "SELECT id, student_id FROM mathe.assessment_4 WHERE student_id IN ('1273')"
                print(p.sql)
                ###############################################################################################

                if p.type == 'Query':
                    pass
                elif p.type == 'Document':
                    match = re.search(r"FROM\s+([^\s]+)", p.sql, re.IGNORECASE)
                    print(match)
                    if match:
                        table_name = match.group(1)
                        base_dir = Path(
                            "/workspaces/Table-Reclamation-Demo/data/mathe_splitted")
                        file_path = base_dir / f"{table_name}.pdf"
                        data_lines = [line for line in texts[j].split(
                            "\n") if line.strip() and table_name not in line]
                        total_dataset_rows += len(data_lines)

                        if file_path.exists():
                            DocumentReader = PdfReader(file_path)
                            for i in range(len(DocumentReader.pages)):
                                texts[j] += DocumentReader.pages[i].extract_text()
                        else:
                            print(f"Error: File not found at {file_path}")
                    else:
                        print(
                            "Error: Could not find a table name in the SQL query.")

                    chunks[j] = chunk_with_header(
                        texts[j], max_tokens=context_size)

                    elapsed_time[j] = time.time()
                    for chunk_idx, chunk in enumerate(chunks[j]):
                        print(
                            f"Processing chunk {chunk_idx}/{len(chunks[j])-1}")

                        response = completion(
                            model=MODEL,
                            messages=[
                                {"role": "system", "content": """
                                    You are a deterministic SQL execution engine.

                                    Your task:
                                    Execute the list of SQL queries EXACTLY on the provided dataset.

                                    --------------------------------
                                    STRICT RULES (MANDATORY)
                                    --------------------------------

                                    1. Output MUST be valid JSON only. No text before or after.
                                    2. Output MUST match EXACTLY this schema:

                                    {
                                    "header": ["column1", "..."],
                                    "data": [["value1", "..."]],
                                    "explanation": "short explanation"
                                    }

                                    3. DO NOT wrap output in a list.
                                    4. DO NOT return multiple JSON objects.
                                    5. DO NOT repeat the query.
                                    6. DO NOT hallucinate values.
                                    7. ONLY return rows that EXACTLY match the WHERE condition.
                                    8. If NO rows match → return:
                                    "data": []

                                    9. Each row in "data" MUST have the SAME number of columns as "header".
                                    10. NEVER return malformed rows (e.g., ["123"] if 2 columns expcted).

                                    --------------------------------
                                    DATASET RULES
                                    --------------------------------

                                    - Dataset is RAW TEXT (space-separated)
                                    - First line = column names
                                    - Each next line = one row
                                    - You MUST manually parse rows
                                    - Columns are separated by spaces

                                    --------------------------------
                                    SQL RULES
                                    --------------------------------

                                    - Only use the provided dataset
                                    - Apply WHERE conditions strictly
                                    - SELECT only requested columns
                                    - DISTINCT = remove duplicates
                                    
                                    IMPORTANT FINAL CHECK (before answering):
                                    - Is JSON valid? ✔
                                    - Does each row match header length? ✔
                                    - Any hallucinated values? ✘
                                    - Any partial matches? ✘

                                    If any rule is violated → FIX before returning.
                                    """},
                                {"role": "user", "content": f"DATASET:{chunk} SQL QUERY: {p.sql} Return ONLY the JSON."}
                            ],
                            response_format=Data,
                            api_base="http://host.docker.internal:11434",
                            timeout=7200,
                        )
                        print(response)
                        all_results.append(response)
                    elapsed_time[j] = time.time() - elapsed_time[j]

            print(all_results)
            walltime = sum(elapsed_time)
            all_data = []
            headers = None

            for res in all_results:
                try:
                    content = res.choices[0].message.content
                    parsed = json.loads(content)
                    if headers is None:
                        headers = parsed.get("header", [])
                    all_data.extend(parsed.get("data", []))
                except Exception as e:
                    print("Skipping invalid response:", e)

            result = {
                "header": headers,
                "data": all_data
            }

            actual_p = 200
            actual_n = max(0, total_dataset_rows - actual_p)

            tp = 0
            fp = 0

            for d in result["data"]:
                if len(d) >= 2 and (str(d[1]) == '3409' or str(d[1]) == '1273'):
                    tp += 1
                else:
                    fp += 1

            fn = max(0, actual_p - tp)
            tn = max(0, actual_n - fp)

            recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
            fall_out = fp / (fp + tn) if (fp + tn) > 0 else 0.0

            f1_denominator = precision + recall
            f1_score = 2 * (precision * recall) / \
                f1_denominator if f1_denominator > 0 else 0.0

            total_population = tp + tn + fp + fn
            accuracy = (tp + tn) / \
                total_population if total_population > 0 else 0.0

            results_log.append({
                "GPU": GPU,
                "context_size": context_size,
                "walltime": walltime,
                "TP": tp,
                "TN": tn,
                "FP": fp,
                "FN": fn,
                "Recall": recall,
                "Precision": precision,
                "Specificity": specificity,
                "Fall-out": fall_out,
                "F1-Score": f1_score,
                "Accuracy": accuracy,
                "num_chunks": len(chunks),
                "num_rows": len(result["data"]),
                "result": result
            })
            print(results_log)

            # --- JSON EXPORT CODE ---
            log_file_path = "logs_postjoined_queries.json"
            existing_logs = {}

            if os.path.exists(log_file_path) and os.path.getsize(log_file_path) > 0:
                try:
                    with open(log_file_path, "r", encoding="utf-8") as f:
                        existing_logs = json.load(f)
                except json.JSONDecodeError:
                    print(
                        f"Warning: '{log_file_path}' contains invalid JSON. Starting fresh.")

            if MODELNAME in existing_logs:
                existing_logs[MODELNAME].extend(results_log)
            else:
                existing_logs[MODELNAME] = results_log

            with open(log_file_path, "w", encoding="utf-8") as f:
                json.dump(existing_logs, f, indent=4)

    # --- PLOT GENERATION OUTSIDE OF EXECUTION LOOP ---
    if results_log:
        recall_list = [r["Recall"] for r in results_log]
        precision_list = [r["Precision"] for r in results_log]
        sizes = [r["context_size"] for r in results_log]

        fig, ax = plt.subplots(figsize=(10, 8))
        ax.plot(
            recall_list,
            precision_list,
            marker='o',
            markersize=8,
            color='tab:purple',
            linestyle='-',
            label='Model Performance Trajectory'
        )

        texts = []
        for idx, txt in enumerate(sizes):
            texts.append(ax.annotate(
                f"{txt}",
                (recall_list[idx], precision_list[idx]),
                fontsize=9,
                bbox=dict(boxstyle="round,pad=0.3",
                          fc="white", ec="gray", alpha=0.8)
            ))

        if HAS_ADJUST_TEXT:
            print("[INFO] Adjusting labels to prevent overlap...")
            adjust_text(
                texts,
                x=recall_list,
                y=precision_list,
                expand_points=(2.0, 2.0),
                force_text=(0.3, 0.6),
                force_points=(0.2, 0.5),
                lim=500,
                arrowprops=dict(arrowstyle="->", color='gray',
                                lw=0.5, alpha=0.7)
            )
        else:
            for t in texts:
                t.set_va('center')
                t.set_ha('center')

        ax.set_title("Precision-Recall Space by Context Size")
        ax.set_xlabel("Recall (True Positive Rate)")
        ax.set_ylabel("Precision (Positive Predictive Value)")
        ax.set_xlim([-0.08, 1.08])
        ax.set_ylim([-0.08, 1.08])
        ax.grid(True, which="both", ls="--", alpha=0.5)
        ax.legend(loc='upper right')

        fig.tight_layout()
        plt.savefig(f"pr_plot_{MODELNAME}.png")
        plt.show()
        plt.close()


@pytest.mark.parametrize("question", QUESTIONS_RAG)
def rag_embedding_insert(question: str):
    base_dir = Path(
        "/workspaces/Table-Reclamation-Demo/data/mathe_unstructured_dataset"
    )
    conn_info = (
        "dbname=rag user=postgres password=password host=db_rag port=5432"
    )

    # 1. Validate directory and look for PDF files
    if not base_dir.exists():
        print(f"Error: Directory not found at {base_dir}")
        return

    pdf_files = list(base_dir.glob("*.pdf"))

    if not pdf_files:
        print(f"No PDF files found in {base_dir}")
        return

    print(f"Found {len(pdf_files)} PDFs to process.")

    # 2. Establish a single database connection for the entire batch
    with psycopg.connect(conn_info) as conn:
        print(f"Database Connection Status: {conn.info.status}")

        with conn.cursor() as cur:
            for file_path in pdf_files:
                source_document = file_path.stem  # Gets filename without .pdf
                print(f"\nProcessing: {file_path.name}...")

                try:
                    # 3. Extract text from the PDF pages
                    reader = PdfReader(file_path)
                    text = "".join(
                        [page.extract_text() or "" for page in reader.pages]
                    ).strip()

                    if not text:
                        print(
                            f"Skipping {file_path.name}: No text could be extracted."
                        )
                        continue

                    # 4. Generate LLM embedding
                    # (Assuming 'embedding' function is imported from your specific library like litellm)
                    response = embedding(
                        model="ollama/embeddinggemma:300m",
                        input=[text],
                        api_base="http://host.docker.internal:11434",
                    )

                    embeddings = response.data[0]["embedding"]

                    # 5. Execute safe parameterized insert query
                    query = """
                        INSERT INTO items (name, embedding)
                        VALUES (%s, %s::vector);                
                    """
                    cur.execute(query, (source_document, embeddings))

                    # Commit after each file to ensure progress is saved if a later file fails
                    conn.commit()
                    print(
                        f"Successfully inserted embeddings for: {source_document}")

                except Exception as e:
                    print(f"Error processing {file_path.name}: {e}")
                    # Rollback the failed transaction chunk so the cursor can keep going
                    conn.rollback()
    print("done with embedding insertion for all documents.")
