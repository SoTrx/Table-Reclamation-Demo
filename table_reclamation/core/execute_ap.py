import duckdb
import os
import pandas as pd


def execute_ap(plan, split_path):
    """
    plan: list of {"table": ..., "sql": ...}
    split_path: path to folder with src_*.csv
    """

    con = duckdb.connect(database=":memory:")

    # 1) register all CSV sources
    csv_files = sorted(
        f for f in os.listdir(split_path)
        if f.endswith(".csv")
    )

    for i, fname in enumerate(csv_files):
        table_name = f"src{i+1}"
        csv_path = os.path.join(split_path, fname)

        con.execute(f"""
            CREATE TEMP TABLE {table_name} AS
            SELECT * FROM read_csv_auto('{csv_path}');
        """)

    # 2) execute AP SQL steps
    results = []

    for step in plan:
        df = con.execute(step["sql"]).fetchdf()
        results.append(df)

    con.close()

    # 3) concat results
    if results:
        return pd.concat(results, ignore_index=True).drop_duplicates()
    else:
        return pd.DataFrame()