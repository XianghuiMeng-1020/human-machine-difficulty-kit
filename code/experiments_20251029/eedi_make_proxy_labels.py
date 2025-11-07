import argparse, csv, os
from collections import defaultdict

def load_scores(path):
    rows=[]
    with open(path, newline="", encoding="utf-8") as f:
        r=csv.DictReader(f)
        for row in r:
            rows.append({
                "question_id": row["question_id"],
                "model": row.get("model",""),
                "correct": int(row["correct"]),
                "p_chosen": float(row["p_chosen"])
            })
    return rows

if __name__=="__main__":
    ap=argparse.ArgumentParser()
    ap.add_argument("--gpt4o_scores", default="data/eedi_gpt4o_300x1/scores.csv")
    ap.add_argument("--mini_scores",  default="data/eedi_gpt4omini_300x1/scores.csv")
    ap.add_argument("--out", default="analysis/eedi_proxy_labels.csv")
    args=ap.parse_args()

    gpt = load_scores(args.gpt4o_scores)
    mini = load_scores(args.mini_scores)

    by_q = defaultdict(dict)
    for r in gpt:
        q = r["question_id"]
        by_q[q]["gpt4o_correct"] = r["correct"]
        by_q[q]["gpt4o_p"] = r["p_chosen"]
    for r in mini:
        q = r["question_id"]
        by_q[q]["mini_correct"] = r["correct"]
        by_q[q]["mini_p"] = r["p_chosen"]

    rows_out=[]
    for qid, d in by_q.items():
        g_ok = d.get("gpt4o_correct",0)
        m_ok = d.get("mini_correct",0)
        g_p  = d.get("gpt4o_p",0.0)
        m_p  = d.get("mini_p",0.0)

        # proxy 规则
        if g_ok==1 and m_ok==1 and g_p>=0.8 and m_p>=0.8:
            h_proxy = "简单H_proxy"
        elif g_ok==0 and m_ok==0 and (g_p>=0.8 or m_p>=0.8):
            h_proxy = "困难H_proxy"
        else:
            h_proxy = "中等H_proxy"

        rows_out.append({
            "question_id": qid,
            "H_proxy": h_proxy,
            "gpt4o_correct": g_ok,
            "gpt4o_p": g_p,
            "mini_correct": m_ok,
            "mini_p": m_p,
        })

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w", newline="", encoding="utf-8") as f:
        hdr=["question_id","H_proxy","gpt4o_correct","gpt4o_p","mini_correct","mini_p"]
        w=csv.DictWriter(f, fieldnames=hdr); w.writeheader(); w.writerows(rows_out)
    print("✅ wrote", args.out)
