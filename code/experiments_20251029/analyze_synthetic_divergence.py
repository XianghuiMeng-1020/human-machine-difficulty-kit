import csv, argparse, os, json
from collections import defaultdict

# 把不同命名的模型归一
ALIASES = {
    "mini": "mini",
    "gpt4o-mini": "mini",
    "gpt4o_min": "mini",
    "qwen": "qwen",
    "qwen3": "qwen",
    "qwen-3-235b": "qwen",
    "deep": "deep",
    "deepseek": "deep",
    "deepseekv3": "deep",
    "gpt4o": "deep",   # 👈 如果你想把 gpt4o 当成“最强”一档，就归到 deep 这组
}

KEEP = {"mini","qwen","deep"}

if __name__=="__main__":
    ap=argparse.ArgumentParser()
    ap.add_argument("--runs", default="synthetic/synthetic_runs.csv")
    ap.add_argument("--outdir", default="analysis/synthetic")
    args=ap.parse_args()

    os.makedirs(args.outdir, exist_ok=True)

    by_q = defaultdict(dict)
    with open(args.runs, newline="", encoding="utf-8") as f:
        r=csv.DictReader(f)
        for row in r:
            qid = row["qid"]
            raw_m = row["model"]
            norm_m = ALIASES.get(raw_m)
            if not norm_m:
                # 不认识的模型名就跳过
                continue
            if norm_m not in KEEP:
                continue
            by_q[qid]["declared_difficulty"] = row["declared_difficulty"]
            by_q[qid]["topic"] = row["topic"]
            by_q[qid][norm_m] = {
                "acc": int(row["correct"]),
                "p": float(row["p_chosen"])
            }

    separators = []
    for qid, d in by_q.items():
        models = [k for k in d.keys() if k in KEEP]
        if not models:
            continue
        accs = {m: d[m]["acc"] for m in models}
        mx = max(accs.values()); mn = min(accs.values())
        # 有的对有的错 → 分离题
        if mx - mn >= 1.0:
            winner = max(accs, key=lambda x: accs[x])
            loser  = min(accs, key=lambda x: accs[x])
            separators.append({
                "qid": qid,
                "declared_difficulty": d.get("declared_difficulty",""),
                "topic": d.get("topic",""),
                "winner": winner,
                "loser": loser,
                "accs": accs
            })

    outpath = os.path.join(args.outdir, "synthetic_separators.json")
    with open(outpath, "w", encoding="utf-8") as f:
        json.dump(separators, f, ensure_ascii=False, indent=2)

    print(f"✅ wrote {outpath} ({len(separators)} separators)")
