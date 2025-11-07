import csv, os

OUT = "paper_assets/mv-hmda/stage1_canonical.csv"

rows = []

# Eedi gpt4o
with open("data/eedi_gpt4o_300x1/scores.csv", newline="", encoding="utf-8") as f:
    r = csv.DictReader(f)
    for row in r:
        rows.append({
            "source": "eedi",
            "qid": row["question_id"],
            "model": "gpt4o",
            "correct": row["correct"],
            "p_chosen": row["p_chosen"],
            "run_id": row["run_id"],
        })

# Eedi mini
with open("data/eedi_gpt4omini_300x1/scores.csv", newline="", encoding="utf-8") as f:
    r = csv.DictReader(f)
    for row in r:
        rows.append({
            "source": "eedi",
            "qid": row["question_id"],
            "model": "gpt4o-mini",
            "correct": row["correct"],
            "p_chosen": row["p_chosen"],
            "run_id": row["run_id"],
        })

os.makedirs(os.path.dirname(OUT), exist_ok=True)
with open(OUT, "w", newline="", encoding="utf-8") as f:
    hdr = ["source","qid","model","correct","p_chosen","run_id"]
    w = csv.DictWriter(f, fieldnames=hdr)
    w.writeheader()
    w.writerows(rows)

print("✅ wrote", OUT, "n=", len(rows))
