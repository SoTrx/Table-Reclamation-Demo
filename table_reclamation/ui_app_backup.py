# if something breaks, delete other files in the facade directory except __init__.py and this file, then run this file instead using "source .venv/bin/activate; streamlit run table_reclamation/ui_app_backup.py".
import json
import os
import sys
import time
from pathlib import Path

import dotenv
import pandas as pd
import psycopg
import streamlit as st
import tiktoken
from core.execute_ap import execute_ap
from core.gen_ap import build_sql_plan, build_storeap_payload, gen_ap_order
from core.nl_to_ur import parse_nl_to_ur
from core.utils import EPrune
from litellm import BaseModel, completion, embedding
from pypdf import PdfReader

# ==========================================
# 1) ENVIRONMENT & GLOBAL CONFIGURATION
# ==========================================
dotenv.load_dotenv()

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

LEXICON_PATH = os.path.join(
    PROJECT_ROOT, "table_reclamation/assets", "lexicon.json")
with open(LEXICON_PATH, "r") as f:
    LEXICON = json.load(f)

# Dual-model setup for cost/compute efficiency
CHEAP_MODEL = "ollama/gemma4:e2b"
EXPENSIVE_MODEL = "ollama/gemma4:31b"
MAX_TOKEN_SIZE = 16384
EMBEDDING_MODEL = "ollama/embeddinggemma:300m"
API_BASE = "http://host.docker.internal:11434"
DB_CONN_INFO = "dbname=rag user=postgres password=password host=db_rag port=5432"

UNSTRUCTURED_DATA_DIR = Path(
    "/workspaces/Table-Reclamation-Demo/data/mathe_unstructured_dataset")
SPLIT_PATH = "./data/MATHE_random_100"

# ==========================================
# 2) PYDANTIC SCHEMAS for LLM Structured Output
# ==========================================


class Data(BaseModel):
    header: list[str]
    data: list[list[str]]
    explanation: str


class SchemaInference(BaseModel):
    inferred_headers: list[str]

# ==========================================
# 3) HELPER FUNCTIONS
# ==========================================


def load_stats(split_path):
    stats_json = os.path.join(split_path, "value_index.json")
    stats_parquet = os.path.join(split_path, "stats.parquet")
    source_files_json = os.path.join(split_path, "source_files.json")

    with open(stats_json, "r") as f:
        value_index = json.load(f)

    df = pd.read_parquet(stats_parquet)
    source_vectors = df.values

    source_files = None
    if os.path.exists(source_files_json):
        with open(source_files_json, "r") as f:
            source_files = json.load(f)

    return {"value_index": value_index, "source_vectors": source_vectors, "source_files": source_files}


def chunk_text(text: str, max_tokens: int, model: str = "gpt-4") -> list[str]:
    enc = tiktoken.encoding_for_model(model)
    lines = text.split("\n")
    chunks, current_chunk = [], []
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
    if not file_path.exists():
        return ""
    try:
        reader = PdfReader(file_path)
        return "".join(page.extract_text() or "" for page in reader.pages)
    except Exception:
        return ""


# ==========================================
# 4) STREAMLIT UI
# ==========================================
st.set_page_config(page_title="TVD Demo", layout="wide")
st.title("Table Reclamation Demo 2026")

# ---- Section 1: Natural Language ----
st.header("1) Natural Language")
nl_query = st.text_area(
    "Query",
    placeholder="Need Discrete Mathematics, Recursivity, level 2",
    height=100
)

# ---- Section 2: Parsed UR ----
st.header("2) Parsed UR")
col1, col2 = st.columns([1, 2])

with col1:
    parse_button = st.button("Parse NL → UR")

if parse_button and nl_query:
    parsed_ur = parse_nl_to_ur(nl_query, LEXICON)
    st.session_state["UR"] = parsed_ur
    # Clear downstream state when a new query is parsed
    for key in ["AP_plan", "rag_mode", "rag_results", "rag_docs", "rag_index"]:
        if key in st.session_state:
            del st.session_state[key]

with col2:
    if "UR" in st.session_state:
        st.code(json.dumps(st.session_state["UR"], indent=2), language="json")
    else:
        st.info("Click 'Parse NL → UR'")

st.divider()

# ---- Section 3: Generate AP / RAG Fallback ----
st.header("3) Generate AP (SQL plan)")

if st.button("Generate AP"):
    if "UR" not in st.session_state:
        st.error("Parse NL → UR first.")
    else:
        UR = st.session_state["UR"]
        stats = load_stats(SPLIT_PATH)
        st.session_state["stats"] = stats

        order = gen_ap_order(UR, stats)
        plan = build_sql_plan(UR, order, stats)

        st.session_state["AP_order"] = order

        # Branch Execution based on Plan content
        if plan:
            st.session_state["AP_plan"] = plan
            st.session_state["rag_mode"] = False
            st.success("SQL Plan generated successfully.")
        else:
            st.session_state["AP_plan"] = []
            st.session_state["rag_mode"] = True
            st.warning("Plan is empty. Falling back to RAG execution.")

            # --- RAG Setup Phase ---
            with st.spinner("Fetching closest documents via pgvector..."):
                try:
                    with psycopg.connect(DB_CONN_INFO) as conn:
                        with conn.cursor() as cur:
                            response = embedding(model=EMBEDDING_MODEL, input=[
                                                 nl_query], api_base=API_BASE)
                            query = """
                                SELECT name, embedding <-> %s::vector AS distance
                                FROM items
                                ORDER BY distance
                                LIMIT 50;
                            """
                            cur.execute(
                                query, (response.data[0]["embedding"],))
                            all_results = cur.fetchall()
                            st.session_state["rag_docs"] = [
                                row[0] for row in all_results] if all_results else []
                except Exception as e:
                    st.error(f"Database connection error: {e}")
                    st.session_state["rag_docs"] = []

            with st.spinner(f"Inferring schema using CHEAP LLM ({CHEAP_MODEL})..."):
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
                            {"role": "user", "content": f"Question: {nl_query}"}
                        ],
                        response_format={
                            "type": "json_object", "schema": SchemaInference.model_json_schema()},
                        api_base=API_BASE,
                        temperature=0,
                    )
                    parsed_schema = json.loads(
                        schema_response.choices[0].message.content)
                    raw_headers = parsed_schema.get("inferred_headers") or []
                    canonical_headers = [h.strip().lower()
                                         for h in raw_headers]
                except Exception as e:
                    st.error(f"Schema inference failed: {e}")
                    canonical_headers = []

                st.session_state["rag_headers"] = canonical_headers
                st.session_state["rag_results"] = []
                st.session_state["rag_index"] = 0


# ==========================================
# 4) EXECUTION BRANCHES
# ==========================================

# BRANCH A: SQL Execution
if "AP_plan" in st.session_state and not st.session_state.get("rag_mode", False):
    st.subheader("SQL Plan")

    for step in st.session_state["AP_plan"]:
        st.markdown(f"**{step['table']}**")
        st.code(step["sql"], language="sql")

    payload = build_storeap_payload(
        nl_query,
        st.session_state["UR"],
        st.session_state["AP_order"],
        st.session_state["AP_plan"],
        dataset="MATHE",
        split="random_100",
    )

    st.subheader("AP Payload (for /storeAP)")
    st.code(json.dumps(payload, indent=2), language="json")

    if st.button("Execute AP"):
        result_df = execute_ap(
            st.session_state["AP_plan"],
            split_path=SPLIT_PATH,
            source_files=st.session_state["stats"].get("source_files")
        )
        st.session_state["result_df"] = result_df

    if "result_df" in st.session_state:
        st.subheader("Execution Result")
        st.dataframe(st.session_state["result_df"], use_container_width=True)

        if st.button("Prune Result"):
            pruned_df = EPrune(
                st.session_state["result_df"], st.session_state["UR"])
            st.session_state["pruned_df"] = pruned_df

    if "pruned_df" in st.session_state:
        st.subheader("Pruned Result")
        st.dataframe(st.session_state["pruned_df"], use_container_width=True)

# BRANCH B: RAG Execution
elif st.session_state.get("rag_mode", False):
    st.subheader("RAG Extraction")

    docs = st.session_state.get("rag_docs", [])
    current_index = st.session_state.get("rag_index", 0)
    headers = st.session_state.get("rag_headers", [])

    st.markdown(f"**Inferred Schema:** `{headers}`")
    st.markdown(f"**Documents Analyzed:** {current_index} / {len(docs)}")

    # Display Accumulated Data
    if st.session_state["rag_results"]:
        st.write("### Extracted Data")
        rag_df = pd.DataFrame(st.session_state["rag_results"], columns=headers)
        st.dataframe(rag_df, use_container_width=True)

    # Interactive Batching UI
    if current_index < len(docs):
        if st.button("Search Next 3 Documents"):
            status_text = st.empty()
            batch = docs[current_index: current_index + 3]
            data_found_in_batch = False

            for i, document in enumerate(batch):
                status_text.info(
                    f"Analyzing document '{document}' with EXPENSIVE LLM ({EXPENSIVE_MODEL})...")

                text = extract_pdf_text(
                    UNSTRUCTURED_DATA_DIR / f"{document}.pdf")
                if not text:
                    continue

                chunks = chunk_text(text, max_tokens=MAX_TOKEN_SIZE)
                document_data, document_headers = [], None

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
                            "header": {json.dumps(headers)},
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
                            {nl_query}

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
                        pass  # Silently skip chunk errors in UI to prevent clutter

                # Align and Verify
                valid_doc_rows = []
                if document_data:
                    for row in document_data:
                        if not all(str(cell).strip().upper() in ("N/A", "", "NONE") for cell in row):
                            row_dict = {h: (row[idx] if idx < len(
                                row) else "N/A") for idx, h in enumerate(document_headers or [])}
                            aligned_row = [row_dict.get(
                                col, "N/A") for col in headers]
                            valid_doc_rows.append(aligned_row)

                if valid_doc_rows:
                    st.session_state["rag_results"].extend(valid_doc_rows)
                    data_found_in_batch = True
                    # Exact Deduplication on the fly
                    st.session_state["rag_results"] = list(
                        {tuple(row) for row in st.session_state["rag_results"]})
                    break  # Short-circuit

            st.session_state["rag_index"] += len(
                batch) if not data_found_in_batch else (i + 1)
            status_text.empty()
            st.rerun()  # Refresh UI to show new data and updated index

    else:
        st.info("Finished analyzing all retrieved documents.")
