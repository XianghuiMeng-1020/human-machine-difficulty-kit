import argparse, json, csv, os
from collections import Counter, defaultdict
import matplotlib.pyplot as plt

def load_sep(path):
    with open(path, encoding="utf-8") as f:
        return json.load(f)

def load_perq(path):
    rows={}
    with open(path, newline="", encoding="utf-8") as f:
        r=csv.DictReader(f)
        for row in r:
            rows[row["qid"]]=row
    return rows

if __name__=="__main__":
    ap=argparse.ArgumentParser()
    ap.add_argument("--perq", default="analysis/race_stage3_per_question.csv")
    ap.add_argument("--sepA", default="analysis/race_stage3_separators_A.json")
    ap.add_argument("--sepB", default="analysis/race_stage3_separators_B.json")
    ap.add_argument("--outdir", default="analysis/race_stage3_report")
    ap.add_argument("--figdir", default="figs/race_stage3_report")
    ap.add_argument("--topk", type=int, default=20)
    args=ap.parse_args()

    os.makedirs(args.outdir, exist_ok=True)
    os.makedirs(args.figdir, exist_ok=True)

    perq = load_perq(args.perq)
    A = load_sep(args.sepA)
    B = load_sep(args.sepB)

    # 1) 赢家计数（谁在分离题上拔尖）
    winA = Counter([x["winner"] for x in A])
    winB = Counter([x["argmax"] for x in B])
    loseA = Counter([x["loser"] for x in A])
    loseB = Counter([x["argmin"] for x in B])

    # 保存计数表
    counts_csv = os.path.join(args.outdir, "separator_counts.csv")
    with open(counts_csv, "w", newline="", encoding="utf-8") as f:
        w=csv.writer(f); w.writerow(["set","model","wins","losses"])
        for m in ["mini","qwen","deep"]:
            w.writerow(["A", m, winA.get(m,0), loseA.get(m,0)])
        for m in ["mini","qwen","deep"]:
            w.writerow(["B", m, winB.get(m,0), loseB.get(m,0)])
    print("✅ wrote", counts_csv)

    # 2) 画柱图（A / B 的胜负次数）
    def bar_counts(win, lose, title, outpng):
        labels=["mini","qwen","deep"]
        x=range(len(labels)); width=0.35
        plt.figure(figsize=(6,4))
        plt.bar([i-width/2 for i in x], [win.get(m,0) for m in labels], width, label="wins")
        plt.bar([i+width/2 for i in x], [lose.get(m,0) for m in labels], width, label="losses")
        plt.xticks(list(x), labels)
        plt.title(title)
        plt.ylabel("count")
        plt.legend()
        plt.tight_layout()
        plt.savefig(outpng, dpi=200)
        print("✅ saved", outpng)

    bar_counts(winA, loseA, "Strong separators A (≥0.8 vs ≤0.2)", os.path.join(args.figdir,"separators_A_wins_losses.png"))
    bar_counts(winB, loseB, "Mid separators B (Δacc≥0.5)",       os.path.join(args.figdir,"separators_B_wins_losses.png"))

    # 3) Top-K 差距最大的题（按 acc_range）
    rows = list(perq.values())
    rows_sorted = sorted(rows, key=lambda r: float(r["acc_range"]), reverse=True)
    topk = rows_sorted[:args.topk]
    top_csv = os.path.join(args.outdir, f"top{args.topk}_by_acc_gap.csv")
    with open(top_csv, "w", newline="", encoding="utf-8") as f:
        hdr=["qid","acc_mini","acc_qwen","acc_deep","p_mini","p_qwen","p_deep","stab_mini","stab_qwen","stab_deep","acc_range","acc_argmax","acc_argmin"]
        w=csv.DictWriter(f, fieldnames=hdr); w.writeheader()
        w.writerows(topk)
    print("✅ wrote", top_csv)

    # 4) 摘要一句话（打印到终端 + 写 md）
    totalA, totalB = len(A), len(B)
    # 哪个模型在 A/B 上 wins 最多
    bestA = max(winA.items(), key=lambda x:x[1])[0] if winA else "N/A"
    bestB = max(winB.items(), key=lambda x:x[1])[0] if winB else "N/A"
    # 哪个模型 losses 最多
    worstA = max(loseA.items(), key=lambda x:x[1])[0] if loseA else "N/A"
    worstB = max(loseB.items(), key=lambda x:x[1])[0] if loseB else "N/A"

    line = (f"RACE 分离题：强分离A共 {totalA} 题，胜场最多的是 {bestA}，失场最多的是 {worstA}；"
            f"中分离B共 {totalB} 题，胜场最多的是 {bestB}，失场最多的是 {worstB}。")
    print("📌", line)

    md = os.path.join(args.outdir, "RESULTS_RACE_STAGE3.md")
    with open(md, "w", encoding="utf-8") as f:
        f.write("# RACE Stage-3 Separators Summary\n\n")
        f.write(f"- Strong A (≥0.8 vs ≤0.2): {totalA} questions\n")
        f.write(f"- Mid B (Δacc≥0.5): {totalB} questions\n\n")
        f.write(f"- A-wins most: **{bestA}**, A-losses most: **{worstA}**\n")
        f.write(f"- B-wins most: **{bestB}**, B-losses most: **{worstB}**\n\n")
        f.write(f"See: `{counts_csv}`, figures in `{args.figdir}`, and top gap list `{top_csv}`.\n")
    print("✅ wrote", md)
