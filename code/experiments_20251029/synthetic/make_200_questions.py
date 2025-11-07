import json, os, random

OUT = "synthetic/gen_questions_200.jsonl"
os.makedirs("synthetic", exist_ok=True)

topics = ["arithmetic", "geometry", "reading", "science", "everyday"]
difficulties = ["easy", "medium", "hard", "adversarial"]

qid = 1
with open(OUT, "w", encoding="utf-8") as f:
    for tp in topics:
        for diff in difficulties:
            # 每个 (topic, diff) 做 10 道 → 5*4*10 = 200
            for k in range(10):
                q = {
                    "qid": f"syn{qid:03d}",
                    "topic": tp,
                    "declared_difficulty": diff,
                    "question": f"({tp}) A synthetic {diff} question number {qid}.",
                    "options": ["A","B","C","D"],
                    "answer": random.choice(["A","B","C","D"])
                }
                f.write(json.dumps(q, ensure_ascii=False) + "\n")
                qid += 1

print("✅ wrote", OUT)
