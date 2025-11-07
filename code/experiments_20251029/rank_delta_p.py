import argparse, json, csv, os
import matplotlib.pyplot as plt

def load_scores(path):
    rows={}
    with open(path, newline="", encoding="utf-8") as f:
        r=csv.DictReader(f)
        for row in r:
            qid=row.get("qid") or row.get("question_id")
            if not qid: continue
            try: p=float(row.get("p_chosen",""))
            except: p=None
            try: c=int(row.get("correct",""))
            except: c=1 if str(row.get("correct","")).lower() in ("1","true","yes") else 0
            rows[str(qid)]={"p":p, "c":c}
    return rows

def collect_rows(qids, label, g_scores, m_scores):
    out=[]
    for q in qids:
        g=g_scores.get(str(q)); m=m_scores.get(str(q))
        if not g or not m: continue
        if g["p"] is None or m["p"] is None: continue
        dp = g["p"] - m["p"]
        da = g["c"] - m["c"]
        out.append({"qid":str(q),"set":label,"p_g":g["p"],"p_m":m["p"],"dp":dp,"acc_g":g["c"],"acc_m":m["c"],"da":da})
    return out

if __name__=="__main__":
    ap=argparse.ArgumentParser()
    ap.add_argument("--mini_hard", required=True)
    ap.add_argument("--mini_easy", required=True)
    ap.add_argument("--gpt4o_scores", default="data/eedi_gpt4o_300x1/scores.csv")
    ap.add_argument("--mini_scores",  default="data/eedi_gpt4omini_300x1/scores.csv")
    ap.add_argument("--out_prefix",   default="analysis/eedi_tau08_deltaP")
    ap.add_argument("--topk", type=int, default=10)
    args=ap.parse_args()

    os.makedirs(os.path.dirname(args.out_prefix), exist_ok=True)

    mini_hard = json.load(open(args.mini_hard, encoding="utf-8"))
    mini_easy = json.load(open(args.mini_easy, encoding="utf-8"))
    g_scores  = load_scores(args.gpt4o_scores)
    m_scores  = load_scores(args.mini_scores)

    rows_h = collect_rows(mini_hard, "miniHard_gptNot", g_scores, m_scores)
    rows_e = collect_rows(mini_easy, "miniEasy_gptNot", g_scores, m_scores)

    # 排序并截取 Top-K by |Δp|
    rows_h_sorted = sorted(rows_h, key=lambda r: abs(r["dp"]), reverse=True)[:args.topk]
    rows_e_sorted = sorted(rows_e, key=lambda r: abs(r["dp"]), reverse=True)[:args.topk]

    # 导出 CSV
    def write_csv(rows, suffix):
        path = f"{args.out_prefix}_{suffix}.csv"
        with open(path, "w", newline="", encoding="utf-8") as f:
            hdr=["qid","set","p_g","p_m","dp","acc_g","acc_m","da"]
            w=csv.DictWriter(f, fieldnames=hdr); w.writeheader(); w.writerows(rows)
        print("✅ wrote", path)

    write_csv(rows_h_sorted, "miniHard_top10")
    write_csv(rows_e_sorted, "miniEasy_top10")

    # 画 Δp 条形图
    def barplot(rows, title, outpng):
        if not rows: 
            print("Empty rows for", title); 
            return
        qids=[r["qid"] for r in rows]
        dps=[r["dp"] for r in rows]
        plt.figure(figsize=(8,4))
        plt.bar(range(len(qids)), dps)
        plt.xticks(range(len(qids)), qids, rotation=45, ha="right")
        plt.ylabel("Δp (gpt4o - mini)")
        plt.title(title)
        plt.tight_layout()
        plt.savefig(outpng, dpi=200)
        print("✅ saved", outpng)

    barplot(rows_h_sorted, "Top-|Δp| (miniHard_gptNot)", f"{args.out_prefix}_miniHard_top10_dp.png")
    barplot(rows_e_sorted, "Top-|Δp| (miniEasy_gptNot)", f"{args.out_prefix}_miniEasy_top10_dp.png")
