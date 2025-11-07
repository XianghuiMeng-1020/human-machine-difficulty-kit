import json, argparse, os
from collections import Counter, defaultdict

if __name__=="__main__":
    ap=argparse.ArgumentParser()
    ap.add_argument("--questions", default="synthetic/gen_questions_200.jsonl")
    ap.add_argument("--separators", default="analysis/synthetic_200/synthetic_separators.json")
    ap.add_argument("--out", default="analysis/synthetic_200/divergence_by_topic.csv")
    args=ap.parse_args()

    # 1) 全部题的 topic
    all_topics = {}
    with open(args.questions, encoding="utf-8") as f:
        for line in f:
            q = json.loads(line)
            all_topics[q["qid"]] = q["topic"]

    # 2) 分离题
    with open(args.separators, encoding="utf-8") as f:
        seps = json.load(f)

    sep_topics = Counter()
    for s in seps:
        qid = s["qid"]
        tp = all_topics.get(qid, "unknown")
        sep_topics[tp] += 1

    # 3) 分母：每个 topic 多少题
    total_topics = Counter(all_topics.values())

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        f.write("topic,total,separators,divergence_rate\n")
        for tp, tot in total_topics.items():
            sep = sep_topics.get(tp, 0)
            rate = sep / tot if tot else 0.0
            f.write(f"{tp},{tot},{sep},{rate:.4f}\n")

    print("✅ wrote", args.out)
