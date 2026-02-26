import streamlit as st
import json, os,sys
import pandas as pd


PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

from demo.gen_ap import build_sql_plan, gen_ap_order,build_storeap_payload
from demo.nl_to_ur import parse_nl_to_ur
import os, json
from demo.execute_ap import execute_ap
from demo.utils import EPrune
LEXICON_PATH = os.path.join(PROJECT_ROOT, "demo", "lexicon.json")
with open(LEXICON_PATH, "r") as f:
    LEXICON = json.load(f)

# Load stats data (value index and source vectors) from the given split path.
def load_stats(split_path):
    stats_json = os.path.join(split_path, "value_index.json")
    stats_parquet = os.path.join(split_path, "stats.parquet")

    with open(stats_json, "r") as f:
        value_index = json.load(f)

    df = pd.read_parquet(stats_parquet)
    source_vectors = df.values
    return {"value_index": value_index, "source_vectors": source_vectors}


st.set_page_config(page_title="TVD Demo", layout="wide")

st.title("Table Reclamation Demo 2026")

# =========================
# 1) Natural Language
# =========================
st.header("1) Natural Language")

nl_query = st.text_area(
    "Query",
    placeholder="Need Discrete Mathematics, Recursivity, level 2",
    height=100
)

# =========================
# 2) Parsed UR
# =========================
st.header("2) Parsed UR")

col1, col2 = st.columns([1, 2])

with col1:
    parse_button = st.button("Parse NL → UR")

if parse_button and nl_query:
    parsed_ur = parse_nl_to_ur(nl_query, LEXICON) 
    
    st.session_state["UR"] = parsed_ur  # to keep the UR saved in the session. 

with col2:
    if "UR" in st.session_state:
        st.code(json.dumps(st.session_state["UR"], indent=2), language="json")
    else:
        st.info("Click 'Parse NL → UR'")

# Divider
st.divider()

# =========================
# 3) Generate AP → Execute → Prune
# =========================
st.header("3) Generate AP (SQL plan)")

SPLIT_PATH = "data/MATHE_random_100"

# ---- STEP 1: Generate AP ----
if st.button("Generate AP"):

    if "UR" not in st.session_state:
        st.error("Parse NL → UR first.")
    else:
        UR = st.session_state["UR"]
        stats = load_stats(SPLIT_PATH)

        order = gen_ap_order(UR, stats)
        plan = build_sql_plan(UR, order, stats)

        st.session_state["AP_order"] = order
        st.session_state["AP_plan"] = plan

# Show AP 
if "AP_plan" in st.session_state:

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

    # ---- STEP 2: Execute AP ----
    if st.button("Execute AP"):

        result_df = execute_ap(
            st.session_state["AP_plan"],
            split_path=SPLIT_PATH
        )

        st.session_state["result_df"] = result_df


# Show execution result 
if "result_df" in st.session_state:

    st.subheader("Execution Result")
    st.dataframe(st.session_state["result_df"], use_container_width=True)

    # ---- STEP 3: Prune ----
    if st.button("Prune Result"):

        pruned_df = EPrune(
            st.session_state["result_df"],
            st.session_state["UR"]
        )

        st.session_state["pruned_df"] = pruned_df


# Show pruned result
if "pruned_df" in st.session_state:

    st.subheader("Pruned Result")
    st.dataframe(st.session_state["pruned_df"], use_container_width=True)