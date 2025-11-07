import argparse, csv, os

def tag_row(prob, corr, tau):
    try:
        p=float(prob)
    except:
        p=0.0
    c=int(corr) if str(corr).isdigit() else (1 if str(corr).lower() in ("true","yes","1") else 0)
    if p>=tau and c==1: return "简单M"
    if p>=tau and c==0: return "困难M"
    return "中等M"

if __name__=="__main__":
    ap=argparse.ArgumentParser()
    ap.add_argument("--scores", required=True)
    ap.add_argument("--tau", type=float, default=0.8)
    ap.add_argument("--out", required=True)
    args=ap.parse_args()

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.scores, newline="", encoding="utf-8") as f, \
         open(args.out, "w", newline="", encoding="utf-8") as g:
        r=csv.DictReader(f)
        hdr=["qid","chosen","p_chosen","correct","M_tau"]
        w=csv.DictWriter(g, fieldnames=hdr); w.writeheader()
        for row in r:
            qid=row.get("qid") or row.get("question_id")
            if not qid: continue
            tag=tag_row(row.get("p_chosen"), row.get("correct"), args.tau)
            w.writerow({
                "qid": qid,
                "chosen": row.get("chosen"),
                "p_chosen": row.get("p_chosen"),
                "correct": row.get("correct"),
                "M_tau": tag
            })
    print("✅ wrote", args.out)
