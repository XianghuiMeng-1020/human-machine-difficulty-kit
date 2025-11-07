import json, os, argparse, random, csv

MODELS = ["mini", "qwen", "deep"]

def fake_answer(model, item):
    base = {"mini":0.55, "qwen":0.7, "deep":0.85}[model]
    if item["declared_difficulty"] == "hard":
        base -= 0.25
    elif item["declared_difficulty"] == "medium":
        base -= 0.1
    # clamp
    base = max(0.05, min(0.95, base))
    # random success
    correct = 1 if random.random() < base else 0
    # fake prob around base
    p = max(0.01, min(0.99, random.gauss(base, 0.05)))
    return correct, p

if __name__=="__main__":
    ap=argparse.ArgumentParser()
    ap.add_argument("--questions", default="synthetic/gen_questions.jsonl")
    ap.add_argument("--out", default="synthetic/synthetic_runs.csv")
    args=ap.parse_args()

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    rows=[]
    for line in open(args.questions, encoding="utf-8"):
        item = json.loads(line)
        for m in MODELS:
            corr, p = fake_answer(m, item)
            rows.append({
                "qid": item["qid"],
                "model": m,
                "declared_difficulty": item["declared_difficulty"],
                "topic": item["topic"],
                "correct": corr,
                "p_chosen": round(p,4)
            })
    with open(args.out, "w", newline="", encoding="utf-8") as f:
        hdr=["qid","model","declared_difficulty","topic","correct","p_chosen"]
        w=csv.DictWriter(f, fieldnames=hdr); w.writeheader(); w.writerows(rows)
    print(f"✅ wrote {args.out}")
