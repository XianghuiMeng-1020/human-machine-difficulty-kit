import csv, argparse
import matplotlib.pyplot as plt

if __name__=="__main__":
    ap=argparse.ArgumentParser()
    ap.add_argument("--metrics_csv", default="analysis/eedi_tau08_divergence_metrics.csv")
    ap.add_argument("--out", default="figs/eedi_tau08_divergence_summary.png")
    args=ap.parse_args()

    rows=[]
    with open(args.metrics_csv, newline="", encoding="utf-8") as f:
        r=csv.DictReader(f)
        for row in r: rows.append(row)

    # 两行（miniHard_gptNot / miniEasy_gptNot），各包含 gpt4o_mean_p / mini_mean_p / gpt4o_acc / mini_acc
    labels=[r["set"] for r in rows]
    g_p=[float(r["gpt4o_mean_p"]) for r in rows]
    m_p=[float(r["mini_mean_p"]) for r in rows]
    g_a=[float(r["gpt4o_acc"]) for r in rows]
    m_a=[float(r["mini_acc"]) for r in rows]

    x=range(len(labels)); width=0.35

    # 置信度条形图
    plt.figure(figsize=(7,4))
    plt.bar([i-width/2 for i in x], g_p, width, label="gpt4o mean p")
    plt.bar([i+width/2 for i in x], m_p, width, label="mini mean p")
    plt.xticks(list(x), labels, rotation=15)
    plt.title("Divergence sets: mean confidence (p_chosen)")
    plt.ylabel("mean p")
    plt.legend()
    plt.tight_layout()
    plt.savefig(args.out.replace(".png","_p.png"), dpi=200)

    # 正确率条形图
    plt.figure(figsize=(7,4))
    plt.bar([i-width/2 for i in x], g_a, width, label="gpt4o acc")
    plt.bar([i+width/2 for i in x], m_a, width, label="mini acc")
    plt.xticks(list(x), labels, rotation=15)
    plt.title("Divergence sets: accuracy")
    plt.ylabel("accuracy")
    plt.legend()
    plt.tight_layout()
    plt.savefig(args.out.replace(".png","_acc.png"), dpi=200)

    print("✅ Saved:", args.out.replace(".png","_p.png"))
    print("✅ Saved:", args.out.replace(".png","_acc.png"))
