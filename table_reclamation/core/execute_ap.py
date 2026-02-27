import os

import duckdb
import pandas as pd


def execute_ap(plan, split_path, source_files=None):
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
        if source_files:
            table_name = source_files[i]   
        else:
            table_name = f"src{i+1}"

        csv_path = os.path.join(split_path, fname)

        # Make reruns safe + handle special chars
        con.execute(f'DROP TABLE IF EXISTS "{table_name}";')
        con.execute(f"""
            CREATE TEMP TABLE "{table_name}" AS
            SELECT * FROM read_csv_auto('{csv_path}');
        """)
    print(con.execute("DESCRIBE src_1").fetchdf())
    print(con.execute("SELECT typeof(newLevel) t, count(*) c FROM src_1 GROUP BY 1").fetchdf())
    print(con.execute("SELECT count(*) FROM src_1 WHERE newLevel='2'").fetchone())
    print(con.execute("SELECT count(*) FROM src_1 WHERE newLevel=2").fetchone()) 

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