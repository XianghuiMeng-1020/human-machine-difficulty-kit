import os, csv, argparse, itertools

if __name__=="__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="data/ednet_sample/KT1")
    ap.add_argument("--out",  default="analysis/ednet_kt1_flat_small.csv")
    ap.add_argument("--max_files", type=int, default=200)
    args = ap.parse_args()

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    files = [f for f in os.listdir(args.root) if f.endswith(".csv")]
    files = sorted(files)[:args.max_files]

    n = 0
    with open(args.out, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["user_id","item_id","correct","timestamp"])
        for fn in files:
            fp = os.path.join(args.root, fn)
            with open(fp, newline="", encoding="utf-8") as ff:
                r = csv.DictReader(ff)
                for row in r:
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
    print("✅ wrote", args.out, "rows:", n, "from files:", len(files))
