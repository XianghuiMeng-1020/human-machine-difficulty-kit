import os
import pandas as pd

RAW = "baselines/bea24_llm_diff/out/all_items_raw.csv"
OUT = "baselines/bea24_llm_diff/out/all_items_raw_filled.csv"

df = pd.read_csv(RAW)

def fill_row(row):
    q = row.get("question", "")
    if isinstance(q, str) and q.strip() != "":
        return q
    src = row.get("source", "")
    qid = str(row.get("qid", ""))
    if src == "eedi":
        # 你以后可以真的去 data/eedi/... 里读题，这里先占位
        return f"Eedi question {qid}: choose the correct option."
    if src == "race":
        # RACE 你现在手里只有 proxy，也先放占位
        return f"RACE question {qid}: reading comprehension multiple-choice."
    if src == "synthetic":
        # synthetic 本来就有题，一般不会走到这里
        return f"Synth question {qid}"
    return f"Generic item {qid}"

df["question"] = df.apply(fill_row, axis=1)

os.makedirs(os.path.dirname(OUT), exist_ok=True)
df.to_csv(OUT, index=False, encoding="utf-8")
print(f"✅ wrote {OUT} rows={len(df)}")
