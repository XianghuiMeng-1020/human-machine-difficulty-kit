import argparse, os
import pandas as pd

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--flat", default="analysis/ednet_flat_u200.csv")
    ap.add_argument("--out", default="baselines/cl4kt_diff/ednet_u200_sessions.csv")
    args = ap.parse_args()

    df = pd.read_csv(args.flat)
    # 期望列：user_id,item_id,correct,timestamp
    needed = {"user_id", "item_id", "correct", "timestamp"}
    if not needed.issubset(df.columns):
        raise ValueError(f"flat file needs columns {needed}, got {df.columns.tolist()}")

    # 按 user 排序 + 合并
    df = df.sort_values(["user_id", "timestamp"])
    rows = []
    for uid, g in df.groupby("user_id"):
        items = " ".join(g["item_id"].astype(str).tolist())
        corrects = " ".join(g["correct"].astype(int).astype(str).tolist())
        rows.append({
            "user_id": uid,
            "items": items,
            "corrects": corrects,
        })

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    pd.DataFrame(rows).to_csv(args.out, index=False)
    print(f"✅ wrote {args.out} rows={len(rows)} from {args.flat}")
