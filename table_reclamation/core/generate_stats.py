import json
import os

import numpy as np
import pandas as pd


def generate_stats_from_folder(folder, store_stem=True):
    
    stats_path = os.path.join(folder, "stats.parquet")
    mapping_path = os.path.join(folder, "value_index.json")
    sources_path = os.path.join(folder, "source_files.json")

    # ---- skip if already computed ----
    if (
        os.path.exists(stats_path)
        and os.path.exists(mapping_path)
        and os.path.exists(sources_path)
    ):
        print("✓ Statistics already exist — skipping generation.")
        return None, None

    csv_files = sorted([f for f in os.listdir(folder) if f.endswith(".csv")])
    sources_list = [pd.read_csv(os.path.join(folder, f), low_memory=False) for f in csv_files]

   # Build value index and source vectors
    value_index = _build_value_index_from_sources(sources_list)

    # Compute source vectors using the value index
    source_vectors = _compute_value_frequencies_from_value_index(sources_list, value_index)

    # Save outputs 
    stats_path = os.path.join(folder, "stats.parquet")
    mapping_path = os.path.join(folder, "value_index.json")

    pd.DataFrame(np.asarray(source_vectors, dtype="float32")).to_parquet(stats_path, index=False)

    value_index_json = {f"{col}:{val}": idx for (col, val), idx in value_index.items()}
    with open(mapping_path, "w") as f:
        json.dump(value_index_json, f)

    # save row order -> file name mapping
    source_files = [os.path.splitext(f)[0] if store_stem else f for f in csv_files]
    with open(os.path.join(folder, "source_files.json"), "w") as f:
        json.dump(source_files, f)

    return value_index, source_vectors


def _build_value_index_from_sources(sources_list):
    # Collect values per column
    col_to_vals = {}
    for df in sources_list:
        for col in df.columns:
            if col not in col_to_vals:
                col_to_vals[col] = set()
            # dropna + unique, then add to set
            for v in df[col].dropna().unique():
                col_to_vals[col].add(v)

    # Deterministic ordering
    value_index = {}
    idx = 0
    for col in sorted(col_to_vals.keys()):
        # sort values deterministically even if mixed types
        vals_sorted = sorted(col_to_vals[col], key=lambda x: str(x))
        for val in vals_sorted:
            value_index[(col, val)] = idx
            idx += 1

    return value_index


def _compute_value_frequencies_from_value_index(sources_list, value_index):
    """
    Same semantics as your original code:
      for each source df:
        vector[i] = count(df[col] == val) / n_rows
    """
    vector_length = len(value_index)
    source_vectors = []

    # Iterate sources
    for df in sources_list:
        vector = np.zeros(vector_length, dtype=np.float32)
        n_rows = len(df)
        if n_rows > 0:
            for (col, val), i in value_index.items():
                if col in df.columns:
                    count = (df[col] == val).sum()
                    vector[i] = count / n_rows
        source_vectors.append(vector)

    return np.array(source_vectors, dtype=np.float32)

def main():

    folder = "/Users/faresa/Desktop/tvd-ap-demo/data/MATHE_random_100" 

    print("\n=== GENERATING STATS ===")
    value_index, vectors = generate_stats_from_folder(folder)

    print("\n=== DONE ===")
    print(f"Number of sources : {len(vectors)}")
    print(f"Vector size       : {len(value_index)}")

    # quick verification
    stats_file = os.path.join(folder, "stats.parquet")
    mapping_file = os.path.join(folder, "value_index.json")
    sources_file = os.path.join(folder, "source_files.json")

    print("\nGenerated files:")
    print(stats_file)
    print(mapping_file)
    print(sources_file)

    df = pd.read_parquet(stats_file)
    print("\nStats shape:", df.shape)


if __name__ == "__main__":
    main()