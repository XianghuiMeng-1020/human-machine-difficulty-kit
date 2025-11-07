import os, csv, argparse

def iter_user_csv(root):
    for fn in os.listdir(root):
        if not fn.endswith(".csv"):
            continue
        fp = os.path.join(root, fn)
        with open(fp, newline="", encoding="utf-8") as f:
            r = csv.DictReader(f)
            for row in r:
                yield row

if __name__=="__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="data/ednet_sample/KT1")
    ap.add_argument("--out",  default="analysis/ednet_kt1_flat.csv")
    args = ap.parse_args()

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    n = 0
    with open(args.out, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["user_id","item_id","correct","timestamp"])
        for row in iter_user_csv(args.root):
            uid = row.get("user_id") or row.get("std_id") or row.get("sid") or ""
            qid = row.get("problem_id") or row.get("item_id") or row.get("question_id") or row.get("cid") or ""
            corr = row.get("correct") or row.get("is_correct") or row.get("right") or "0"
            ts   = row.get("timestamp") or row.get("ts") or row.get("start_time") or ""
            if not qid:
                continue
            try:
                corr = int(corr)
            except Exception:
                corr = 1 if str(corr).lower() in ("true","t","yes") else 0
            w.writerow([uid, qid, corr, ts])
            n += 1
    print("✅ wrote", args.out, "rows:", n)
