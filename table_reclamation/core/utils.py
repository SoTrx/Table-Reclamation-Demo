import pandas as pd
from collections import defaultdict


def EPrune(T, UR):
    T = T.copy()

    UR_sets = {col: set(vals) for col, vals in UR.items()}
    cols = [col for col in UR_sets.keys() if col in T.columns]
    if not cols or T.empty:
        return T

    # mask[col] = rows where T[col] is a UR value
    masks = {col: T[col].isin(UR_sets[col]) for col in cols}

    # same "_matches" meaning as your old code
    T["_S_size"] = pd.DataFrame(masks).sum(axis=1)

    # counters count[(col,val)] = how many rows cover that UR value
    count = defaultdict(int)
    for col in cols:
        vc = T.loc[masks[col], col].value_counts(dropna=False)
        for v, c in vc.items():
            if pd.isna(v):
                continue
            count[(col, v)] += int(c)

    # keep your "old" sorting rule (same line)
    T_sorted = T.sort_values("_S_size", ascending=True, kind="mergesort")


    to_drop = []
    for idx in T_sorted.index:
        items = []
        for col in cols:
            if masks[col].at[idx]:
                v = T.at[idx, col]
                if not pd.isna(v):
                    items.append((col, v))

        if not items:
            to_drop.append(idx)
            continue

        if all(count[it] > 1 for it in items):
            to_drop.append(idx)
            for it in items:
                count[it] -= 1

    return T.drop(index=to_drop).drop(columns="_S_size")



