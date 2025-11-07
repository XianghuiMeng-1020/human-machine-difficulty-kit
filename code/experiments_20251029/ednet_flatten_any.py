import os, csv, argparse
from glob import glob

def flatten(root, out, max_files=None):
    files = sorted(glob(os.path.join(root, "*.csv")))
    if max_files:
        files = files[:max_files]
    os.makedirs(os.path.dirname(out), exist_ok=True)
    nrows = 0
    with open(out, "w", newline="", encoding="utf-8") as f_out:
        w = csv.writer(f_out)
        w.writerow(["user_id","item_id","correct","timestamp"])
        for i, fp in enumerate(files):
            with open(fp, newline="", encoding="utf-8") as f_in:
                r = csv.DictReader(f_in)
                for row in r:
                    w.writerow([
                        os.path.splitext(os.path.basename(fp))[0],  # user_id from filename
                        row.get("problem_id") or row.get("item_id") or row.get("question_id"),
                        row.get("correct") or row.get("solved") or "0",
                        row.get("timestamp") or row.get("time") or ""
                    ])
                    nrows += 1
    print(f"✅ wrote {out} rows: {nrows} from files: {len(files)}")

if __name__=="__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--max_files", type=int, default=None)
    args = ap.parse_args()
    flatten(args.root, args.out, args.max_files)
