import os
import pandas as pd

RAW_PATH = "baselines/bea24_llm_diff/out/all_items_raw.csv"
OUT_DIR = "baselines/bea24_llm_diff/out"
os.makedirs(OUT_DIR, exist_ok=True)

df = pd.read_csv(RAW_PATH)
cols = set(df.columns)

def get_item_id(row, idx):
    if "qid" in cols:
        return str(row.qid)
    if "question_id" in cols:
        return str(row.question_id)
    if "id" in cols:
        return str(row.id)
    return f"item_{idx}"

def get_question(row):
    if "question" in cols and pd.notna(row.question):
        return str(row.question)
    return "PLACEHOLDER: question text not available"

def get_label(row):
    if "H_proxy" in cols and pd.notna(row.H_proxy):
        return str(row.H_proxy)
    if "label" in cols and pd.notna(row.label):
        return str(row.label)
    return "中等H_proxy"

rows = []
for idx, row in enumerate(df.itertuples(index=False)):
    rows.append({
        "item_id": get_item_id(row, idx),
        "question": get_question(row),
        "label": get_label(row),
        "source": getattr(row, "source", "unknown"),
    })

out_csv = os.path.join(OUT_DIR, "all_items_bea_like.csv")
pd.DataFrame(rows).to_csv(out_csv, index=False, encoding="utf-8")
print(f"✅ wrote {out_csv} rows={len(rows)}")
print("👉 如果你之后把 Eedi / RACE 真题面补上，再跑一遍这个脚本就能覆盖。")
