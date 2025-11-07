import argparse, os
import pandas as pd
from collections import defaultdict

"""
输入：你前面已经有的这些文件，比如：
  - analysis/ednet_flat_u200.csv
  - analysis/ednet_flat_u500.csv
  - analysis/ednet_flat_u1000.csv

这些文件列应该是：user_id,item_id,correct,timestamp
我们把它们聚合成每个 user 的时间序列，存成一个 csv，后面训练就能直接 load。
"""

def build_sessions(df: pd.DataFrame):
    df = df.dropna(subset=["user_id", "item_id"])
    df = df.sort_values(["user_id", "timestamp"])
    grouped = defaultdict(list)
    for row in df.itertuples(index=False):
        grouped[row.user_id].append((row.item_id, int(row.correct)))
    return grouped

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--logs", required=True, help="flattened ednet csv, e.g. analysis/ednet_flat_u200.csv")
    ap.add_argument("--out", required=True, help="output sessions csv")
    ap.add_argument("--min_len", type=int, default=5, help="min sequence length to keep")
    args = ap.parse_args()

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    df = pd.read_csv(args.logs)
    sessions = build_sessions(df)

    # 展开成一行一个用户，序列用空格拼
    rows = []
    for uid, seq in sessions.items():
        if len(seq) < args.min_len:
            continue
        items = " ".join([str(it) for it, c in seq])
        corrs = " ".join([str(c) for it, c in seq])
        rows.append({"user_id": uid, "items": items, "corrects": corrs, "seq_len": len(seq)})

    out_df = pd.DataFrame(rows)
    out_df.to_csv(args.out, index=False)
    print(f"✅ wrote {args.out} rows={len(out_df)} (min_len={args.min_len})")
