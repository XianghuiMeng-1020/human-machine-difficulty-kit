import os, json, csv, argparse, time, random

# 你们自己的模型名字 / endpoint
MODELS = [
    # (id, display_name)
    ("gpt4o",  "gpt-4o"),
    ("gpt4o-mini", "gpt-4o-mini"),
    ("qwen3",  "qwen-3-235b"),
]

def call_model(model_name: str, question: dict) -> tuple[int, float]:
    """
    这里换成你们的真实调用。
    要求返回: (correct, p_chosen)
    - correct: 1/0
    - p_chosen: 0~1 的概率/置信度
    下面是一个 stub，先随机，方便你本地试
    """
    base = 0.7
    if "hard" in question.get("declared_difficulty",""):
        base -= 0.25
    if model_name.endswith("mini"):
        base -= 0.15
    base = max(0.05, min(0.95, base))
    ok = 1 if random.random() < base else 0
    p  = max(0.01, min(0.99, random.gauss(base, 0.06)))
    return ok, p

if __name__=="__main__":
    ap=argparse.ArgumentParser()
    ap.add_argument("--questions", default="synthetic/gen_questions.jsonl")
    ap.add_argument("--out", default="runs/synthetic_llm_runs.csv")
    ap.add_argument("--sleep", type=float, default=0.0, help="sleep between calls (s)")
    args=ap.parse_args()

    os.makedirs(os.path.dirname(args.out), exist_ok=True)

    rows=[]
    with open(args.questions, encoding="utf-8") as f:
        for line in f:
            q = json.loads(line)
            for mid, mdisp in MODELS:
                ok, prob = call_model(mid, q)
                rows.append({
                    "qid": q["qid"],
                    "model": mid,
                    "declared_difficulty": q.get("declared_difficulty",""),
                    "topic": q.get("topic",""),
                    "correct": ok,
                    "p_chosen": round(prob,4)
                })
                if args.sleep>0:
                    time.sleep(args.sleep)

    with open(args.out, "w", newline="", encoding="utf-8") as f:
        hdr=["qid","model","declared_difficulty","topic","correct","p_chosen"]
        w=csv.DictWriter(f, fieldnames=hdr); w.writeheader(); w.writerows(rows)

    print(f"✅ wrote {args.out}")
