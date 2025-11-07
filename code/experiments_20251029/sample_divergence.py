import argparse, json, csv, random, os
import matplotlib.pyplot as plt

def load_scores(path):
    rows={}
    with open(path, newline="", encoding="utf-8") as f:
        r=csv.DictReader(f)
        for row in r:
            qid=row.get("qid") or row.get("question_id")
            if qid: rows[str(qid)]=row
    return rows

if __name__=="__main__":
    ap=argparse.ArgumentParser()
    ap.add_argument("--mini_hard", required=True)  # mini=困难M & gpt4o∈{简单/中等}
    ap.add_argument("--mini_easy", required=True)  # mini=简单M & gpt4o∈{困难/中等}
    ap.add_argument("--gpt4o_scores", default="data/eedi_gpt4o_300x1/scores.csv")
    ap.add_argument("--mini_scores",  default="data/eedi_gpt4omini_300x1/scores.csv")
    ap.add_argument("--gpt4o_tags",   default="analysis/eedi_gpt4o_tau08_model_tags.csv")
    ap.add_argument("--mini_tags",    default="analysis/eedi_gpt4omini_tau08_model_tags.csv")
    ap.add_argument("--out_prefix",   default="analysis/eedi_tau08_divergence")
    ap.add_argument("--k", type=int, default=10)
    args=ap.parse_args()

    os.makedirs(os.path.dirname(args.out_prefix), exist_ok=True)

    # 加载清单
    with open(args.mini_hard,encoding="utf-8") as f: mini_hard = json.load(f)
    with open(args.mini_easy,encoding="utf-8") as f: mini_easy = json.load(f)

    # 抽样
    random.seed(42)
    pick_hard = mini_hard[:args.k] if len(mini_hard)<=args.k else random.sample(mini_hard, args.k)
    pick_easy = mini_easy[:args.k] if len(mini_easy)<=args.k else random.sample(mini_easy, args.k)

    # 加载 per-question 分数与标签
    g_scores = load_scores(args.gpt4o_scores)
    m_scores = load_scores(args.mini_scores)

    def load_tags(path):
        d={}
        with open(path, newline="", encoding="utf-8") as f:
            r=csv.DictReader(f)
            for row in r:
                qid=row.get("qid") or row.get("question_id")
                if qid: d[str(qid)]=row.get("M_tau")
        return d
    g_tags = load_tags(args.gpt4o_tags)
    m_tags = load_tags(args.mini_tags)

    # 组装对比表
    def assemble(qids, label):
        out=[]
        for q in qids:
            g = g_scores.get(str(q), {})
            m = m_scores.get(str(q), {})
            out.append({
                "qid": q,
                "set": label,
                "gpt4o_M": g_tags.get(str(q),""),
                "gpt4o_p": g.get("p_chosen",""),
                "gpt4o_correct": g.get("correct",""),
                "mini_M": m_tags.get(str(q),""),
                "mini_p": m.get("p_chosen",""),
                "mini_correct": m.get("correct",""),
            })
        return out

    rows = assemble(pick_hard, "miniHard_gptNot") + assemble(pick_easy, "miniEasy_gptNot")

    # 输出 CSV
    table_path = args.out_prefix + "_sample_table.csv"
    with open(table_path, "w", newline="", encoding="utf-8") as f:
        hdr=["qid","set","gpt4o_M","gpt4o_p","gpt4o_correct","mini_M","mini_p","mini_correct"]
        w=csv.DictWriter(f, fieldnames=hdr); w.writeheader(); w.writerows(rows)
    print("✅ wrote", table_path)

    # 小柱图：两类分歧数量
    counts = {"miniHard_gptNot": len(mini_hard), "miniEasy_gptNot": len(mini_easy)}
    plt.figure()
    plt.bar(list(counts.keys()), list(counts.values()))
    plt.title("Counts of divergence cases (τ=0.8)")
    plt.ylabel("count")
    plt.tight_layout()
    fig_path = args.out_prefix + "_divergence_counts.png"
    plt.savefig(fig_path, dpi=180)
    print("✅ saved", fig_path)
