import os, json
import pandas as pd

IN_CSV = "baselines/bea24_llm_diff/out/all_items_bea_like.csv"
OUT_CSV = "baselines/bea24_llm_diff/out/all_items_bea_filled.csv"

# 1) 尝试加载你已有的 RACE 原始题面（如果没有，就用 placeholder）
RACE_TXT = {}
race_src_paths = [
    "data/race/middle.jsonl",
    "data/race/high.jsonl",
    "data/race/all.jsonl",
]
for p in race_src_paths:
    if os.path.exists(p):
        with open(p, "r", encoding="utf-8") as f:
            for line in f:
                obj = json.loads(line)
                qid = obj.get("id") or obj.get("example_id") or obj.get("qid")
                if not qid:
                    continue
                RACE_TXT[str(qid)] = obj.get("article", "") + "\n" + obj.get("question", "")

# 2) 尝试加载 Eedi 题面（你有就放 data/eedi/items.jsonl）
EEDI_TXT = {}
eedi_path = "data/eedi/items.jsonl"
if os.path.exists(eedi_path):
    with open(eedi_path, "r", encoding="utf-8") as f:
        for line in f:
            obj = json.loads(line)
            qid = str(obj.get("question_id") or obj.get("id"))
            stem = obj.get("stem") or obj.get("question") or ""
            EEDI_TXT[qid] = stem

df = pd.read_csv(IN_CSV)

filled_questions = []
for _, row in df.iterrows():
    src = row.get("source", "unknown")
    cur_q = row.get("question", "")
    if isinstance(cur_q, str) and cur_q.strip():
        filled_questions.append(cur_q)
        continue

    item_id = str(row.get("item_id", "NA"))
    if src == "race" and item_id in RACE_TXT:
        filled_questions.append(RACE_TXT[item_id])
    elif src == "eedi" and item_id in EEDI_TXT:
        filled_questions.append(EEDI_TXT[item_id])
    else:
        filled_questions.append(f"Generic item {item_id} (placeholder).")

df["question"] = filled_questions
os.makedirs(os.path.dirname(OUT_CSV), exist_ok=True)
df.to_csv(OUT_CSV, index=False, encoding="utf-8")
print(f"✅ wrote {OUT_CSV} rows={len(df)}")
