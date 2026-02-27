# Table Reclamation Demo

## Overview

This repository contains a standalone demo of a **Stats-Guided Analytical Pattern (AP) generation system** for structured data discovery.

The pipeline:

**Natural Language → Structured UR → Stat-Guided SQL Plan → Execution → Pruning**

This demo operates over split versions of the MATHE dataset.

---

## Project Structure

```
table_reclamation/
  __init__.py
  ui_app.py
  assets/
    lexicon.json
  core/
    __init__.py
    execute_ap.py
    gen_ap.py
    nl_to_ur.py
    utils.py
  domain/
    __init__.py
    sql_operation.py
  facade/
    __init__.py
    table_reclamation.py

data/
  MATHE_random_100/
    src_*.csv
    stats.parquet
    value_index.json
```

---

## Setup

### Install dependencies

```bash
# Library only
uv sync

# With the Streamlit UI
uv sync --extra ui
```

---

## Run the Demo UI

```bash
uv run streamlit run table_reclamation/ui_app.py
```

Open the local URL in your browser.

---

## Library Usage

```python
from pathlib import Path
from table_reclamation import AccessPlanner

planner = AccessPlanner(tables_path=Path("data/MATHE_random_100"))
plan = planner.generate_plan("Find level 3 questions on algebra")
for op in plan:
    print(op)
```

Or run a script directly with uv:

```bash
uv run python my_script.py
```

---

## Dataset

- Dataset: MATHE
- Split: random_100
- Sources stored as CSV
- Statistics stored in Parquet and JSON

Execution runs fully in-memory using DuckDB.

---

## AP Payload (PGJSON)

Each generated Analytical Pattern contains:

- `nl` – original natural language query
- `ur` – structured User Request
- `source_order` – ordered selected sources
- `sql_plan` – executable SQL steps
- `meta` – dataset, split, method, timestamp

---

## Author

Ahmad Fares
2026

---
