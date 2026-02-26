import json, time, os
import pandas as pd
def _stats_keys(col, v):
    if isinstance(v, int):
        return [f"{col}:{v}", f"{col}:{float(v)}", f"{col}:{str(v)}"]
    if isinstance(v, float):
        iv = int(v)
        return [f"{col}:{v}", f"{col}:{iv}", f"{col}:{str(v)}", f"{col}:{str(iv)}"]
    return [f"{col}:{v}", f"{col}:{str(v)}"]
# Example: _stats_keys("question_id", 80) -> ["question_id:80", "question_id:80.0", "question_id:'80'"]
def _sql_literal(v):
    if isinstance(v, str):
        return "'" + v.replace("'", "''") + "'"
    return str(v)


# Given a UR and stats data, determine a good order of sources to cover the UR. This is a greedy algorithm that at each step picks the source that covers the largest number of remaining UR values.
def gen_ap_order(UR, stats_data):
    value_index = stats_data["value_index"]          
    source_vectors = stats_data["source_vectors"]   

    remaining = {c: set(vs) for c, vs in UR.items()}
    order = []

    n_sources = source_vectors.shape[0]

    while any(remaining.values()):
        best_src = None
        best_gain = 0
        best_cover = {}

        for src_idx in range(n_sources):
            if src_idx in order:
                continue

            vec = source_vectors[src_idx]
            gain = 0
            cover = {}

            for col, vals in remaining.items():
                for v in vals:
                    j = None
                    for key in _stats_keys(col, v):
                        j = value_index.get(key)
                        if j is not None:
                            break
                    if j is None:
                        continue
                    if vec[j] > 0:
                        gain += 1
                        cover.setdefault(col, set()).add(v)

            if gain > best_gain:
                best_gain = gain
                best_src = src_idx
                best_cover = cover

        if best_src is None or best_gain == 0:
            break

        order.append(best_src)

        for col, covered_vals in best_cover.items():
            remaining[col] -= covered_vals

    return order

# Given a UR, an order of sources, and stats data, build a SQL plan that covers the UR. The plan is a list of steps, where each step specifies a source and a SQL query that retrieves the relevant rows from that source.
def build_sql_plan(UR, order, stats_data, table_prefix="src"):
        value_index = stats_data["value_index"]
        source_vectors = stats_data["source_vectors"]

        remaining = {c: set(vs) for c, vs in UR.items()}
        plan = []

        for src_idx in order:
            vec = source_vectors[src_idx]
            conditions = []
            covered_now = {}

            for col, vals in remaining.items():
                hits = []
                for v in vals:
                    j = None
                    for key in _stats_keys(col, v):
                        j = value_index.get(key)
                        if j is not None:
                            break
                    if j is None:
                        continue
                    if j is not None and vec[j] > 0:
                        hits.append(v)
                    if j is None:
                        print("MISSING:", col, v)
                   

                if hits:
                    covered_now[col] = set(hits)
                    hits = sorted(hits, key=str)

                    
                    in_list = ", ".join(_sql_literal(x) for x in hits)
                    conditions.append(f"{col} IN ({in_list})")

            if conditions:
                tbl = f"{table_prefix}{src_idx+1}"
                # sql = f"SELECT * FROM {tbl} WHERE " + " OR ".join(conditions)
                
                # In this case, we select only the relevant columns, the ones found in the UR: if the whole result needed, use the commented line above instead.
                cols = ", ".join(UR.keys())
                sql = f"SELECT DISTINCT {cols} FROM {tbl} WHERE " + " OR ".join(conditions)
                plan.append({"src_idx": src_idx, "table": tbl, "sql": sql})

                for col, vs in covered_now.items():
                    remaining[col] -= vs

            if not any(remaining.values()):
                break

        return plan
    
   
# Build the payload for the /storeAP endpoint, which includes the original NL query, the parsed UR, the source order, and the generated SQL plan. This will be stored in Neo4j as a PGJSON object.
def build_storeap_payload(nl_query, UR, order, plan, *, dataset="MATHE", split="random_100"):
    ap_obj = {
        "nl": nl_query,
        "ur": UR,
        "source_order": [f"src{i+1}" for i in order],
        "sql_plan": plan,
        "meta": {
            "dataset": dataset,
            "split": split,
            "method": "AM",
            "stats_guided": True,
            "timestamp": int(time.time()),
        },
    }
    return {"ap": json.dumps(ap_obj)}

def load_stats(split_path):
        stats_json = os.path.join(split_path, "value_index.json")
        stats_parquet = os.path.join(split_path, "stats.parquet")

        with open(stats_json, "r") as f:
            value_index = json.load(f)

        df = pd.read_parquet(stats_parquet)
        source_vectors = df.values
        return {"value_index": value_index, "source_vectors": source_vectors}



# The following is a simple test/demo of the above functions. You can run this file directly to see how it works, and to generate example SQL plans for given URs.    
if __name__ == "__main__":

    import os, json


    split_path = "/Users/faresa/Desktop/TVD/data/generated_splits/MATHE/random_100"  
    stats = load_stats(split_path)

    # UR = {
    #         "keyword_name": ["Cauchy problem"],
    #         "topic_name": ["Integration", "Discrete Mathematics"],
    #         "subtopic_name": ["Recursivity"],
    #         "question_id": [80],
    #     }
    
    UR = {
            "topic_name": [
                "Optimization",
                "Linear Algebra",
                "Fundamental Mathematics",
                "Differentiation"
            ],
            "subtopic_name": [
                "Nonlinear Optimization",
                "Vector Spaces",
                "Derivatives",
                "Algebraic expressions, Equations, and Inequalities"
            ],
            "keyword_name": [
                "Optimal solution",
                "Vector subspace",
                "Geometric transformations",
                "Product rule",
                "Simplify expressions"
            ],
            "newLevel": [
                "4",
                "2",
                "3",
                "1"
            ],
            "algorithmLevel": [
                "4",
                "1",
                "3"
            ],
            "question_id": [
                "859",
                "457",
                "504",
                "96",
                "614"
            ],
            "id_lect": [
                "61",
                "81",
                "91",
                "39",
                "50"
            ]
            }

    order = gen_ap_order(UR, stats)
    plan = build_sql_plan(UR, order, stats)
    for step in plan:
        print(step["sql"])
  
    
    