import csv, os, glob

OUT = "paper_assets/mv-hmda_race/stage1_canonical_race.csv"
os.makedirs(os.path.dirname(OUT), exist_ok=True)

rows = []

# 我们知道你这三个目录都在 outputs/ 下面
SOURCES = [
    ("gpt4omini", "outputs/race_gpt4omini_600x5/scores.csv"),
    ("qwen3next80b", "outputs/race_qwen3next80b_600x5/scores.csv"),
    ("deepseekv3", "outputs/race_deepseekv3_600x5/scores.csv"),
]

for model_id, path in SOURCES:
    if not os.path.exists(path):
        print("⚠️ missing", path)
        continue
    with open(path, newline="", encoding="utf-8") as f:
        r = csv.DictReader(f)
        for row in r:
            # RACE 的 scores.csv 里有 question_id, run_id, correct, p_chosen ...
            rows.append({
                "source": "race",
                "qid": row["question_id"],
                "model": model_id,
                "correct": row["correct"],
                "p_chosen": row["p_chosen"],
                "run_id": row.get("run_id",""),
            })

with open(OUT, "w", newline="", encoding="utf-8") as f:
    hdr = ["source","qid","model","correct","p_chosen","run_id"]
    w = csv.DictWriter(f, fieldnames=hdr)
    w.writeheader()
    w.writerows(rows)

print("✅ wrote", OUT, "n=", len(rows))
