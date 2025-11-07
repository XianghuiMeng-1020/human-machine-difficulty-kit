import os, argparse, pandas as pd
from collections import defaultdict

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--logs", default="analysis/ednet_flat_ednet_true.csv")
    ap.add_argument("--out_counts", default="analysis/ednet_item_answer_counts.csv")
    ap.add_argument("--chunksize", type=int, default=500_000)
    args = ap.parse_args()

    counts = defaultdict(int)
    total_rows = 0

    for chunk in pd.read_csv(args.logs, chunksize=args.chunksize):
        # 期望列：user_id,item_id,student_answer,gold_answer,correct,timestamp
        for row in chunk.itertuples(index=False):
            item = str(row.item_id)
            ans  = str(row.student_answer)
            counts[(item, ans)] += 1
        total_rows += len(chunk)
        print(f"[pass] processed {total_rows} rows ...")

    # 写出成 CSV
    rows = []
    for (item, ans), cnt in counts.items():
        rows.append({"item_id": item, "student_answer": ans, "cnt": cnt})

    os.makedirs(os.path.dirname(args.out_counts), exist_ok=True)
    pd.DataFrame(rows).to_csv(args.out_counts, index=False)
    print(f"✅ wrote {args.out_counts} rows={len(rows)} from={args.logs}")
