import argparse, json, csv, os
from statistics import mean

def load_scores(path):
    rows={}
    with open(path, newline="", encoding="utf-8") as f:
        r=csv.DictReader(f)
        for row in r:
            qid=row.get("qid") or row.get("question_id")
            if not qid: continue
            try:
                p=float(row.get("p_chosen", ""))
            except:
                p=None
            try:
                c=int(row.get("correct", ""))
            except:
                c=1 if str(row.get("correct","")).lower() in ("1","true","yes") else 0
            rows[str(qid)]={"p":p, "c":c}
    return rows

def stats_for(qids, a_scores, b_scores):
    Aps, Bps, Acs, Bcs = [], [], [], []
    for q in qids:
        qa, qb = a_scores.get(str(q)), b_scores.get(str(q))
        if not qa or not qb: continue
        if qa["p"] is not None: Aps.append(qa["p"])
        if qb["p"] is not None: Bps.append(qb["p"])
        Acs.append(qa["c"]); Bcs.append(qb["c"])
    n = min(len(Acs), len(Bcs))
    mAp = mean(Aps) if Aps else 0.0
    mBp = mean(Bps) if Bps else 0.0
    mAc = mean(Acs) if Acs else 0.0
    mBc = mean(Bcs) if Bcs else 0.0
    return n, mAp, mBp, mAc, mBc

if __name__=="__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--mini_hard", required=True)  # mini=困难M & gpt4o∈{简单/中等}
    ap.add_argument("--mini_easy", required=True)  # mini=简单M & gpt4o∈{困难/中等}
    ap.add_argument("--gpt4o_scores", default="data/eedi_gpt4o_300x1/scores.csv")
    ap.add_argument("--mini_scores",  default="data/eedi_gpt4omini_300x1/scores.csv")
    ap.add_argument("--out", default="analysis/eedi_tau08_divergence_metrics.csv")
    args = ap.parse_args()

    os.makedirs(os.path.dirname(args.out), exist_ok=True)

    mini_hard = json.load(open(args.mini_hard, encoding="utf-8"))
    mini_easy = json.load(open(args.mini_easy, encoding="utf-8"))
    g_scores  = load_scores(args.gpt4o_scores)
    m_scores  = load_scores(args.mini_scores)

    rows=[]
    for name, qids in [("miniHard_gptNot", mini_hard), ("miniEasy_gptNot", mini_easy)]:
        n, g_p, m_p, g_a, m_a = stats_for(qids, g_scores, m_scores)
        rows.append({
            "set": name, "n": n,
            "gpt4o_mean_p": round(g_p,4), "mini_mean_p": round(m_p,4), "delta_p(gpt4o-mini)": round(g_p-m_p,4),
            "gpt4o_acc": round(g_a,4),    "mini_acc": round(m_a,4),    "delta_acc(gpt4o-mini)": round(g_a-m_a,4)
        })

    # 保存
    with open(args.out, "w", newline="", encoding="utf-8") as f:
        hdr=["set","n","gpt4o_mean_p","mini_mean_p","delta_p(gpt4o-mini)","gpt4o_acc","mini_acc","delta_acc(gpt4o-mini)"]
        w=csv.DictWriter(f, fieldnames=hdr); w.writeheader(); w.writerows(rows)

    print("✅ wrote", args.out)
    for r in rows:
        s=r["set"]; n=r["n"]; dp=r["delta_p(gpt4o-mini)"]; da=r["delta_acc(gpt4o-mini)"]
        print(f"[{s}] n={n} | Δp={dp:+.3f} | Δacc={da:+.3f}")
        if s=="miniHard_gptNot":
            print("  → 小模型判“难”而大模型不难：若 Δacc>0 / Δp>0，说明 gpt4o 更强且更自信；反之提示 mini 的‘谨慎’并非多余。")
        else:
            print("  → 小模型判“易”而大模型不易：若 Δacc<0 / Δp<0，说明 mini 的‘自信’多数为错自信；若 Δacc>0 可能是题型捷径。")
