import json, os, argparse, random

TOPICS = [
    "reading-comprehension:inference",
    "reading-comprehension:detail",
    "vocab-in-context",
    "math:ratio",
    "math:geometry",
    "science:reasoning"
]
DIFFS = ["easy","medium","hard"]

TEMPL = "You are a teacher. Generate a {diff} {topic} multiple-choice question with 4 options and indicate the correct option."

if __name__=="__main__":
    ap=argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=30)
    ap.add_argument("--out", default="synthetic/gen_questions.jsonl")
    args=ap.parse_args()

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        for i in range(args.n):
            topic = random.choice(TOPICS)
            diff  = random.choice(DIFFS)
            item = {
                "qid": f"syn_{i:03d}",
                "topic": topic,
                "declared_difficulty": diff,
                "generator_prompt": TEMPL.format(diff=diff, topic=topic),
                "problem_text": f"[STUB] {topic} / {diff} / this is placeholder question {i}",
                "options": ["A","B","C","D"],
                "answer": random.choice(["A","B","C","D"])
            }
            f.write(json.dumps(item, ensure_ascii=False) + "\n")
    print(f"✅ wrote {args.out}")
