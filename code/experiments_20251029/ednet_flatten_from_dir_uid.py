import os, argparse, pandas as pd

def detect_cols(df):
    # KT1 常见字段名兜一遍
    cand_item = ["problem_id", "question_id", "item_id"]
    cand_corr = ["correct", "is_correct", "answerCode"]
    cand_time = ["timestamp", "ts", "start_timestamp", "elapsed_time"]

    item_col = None
    for c in cand_item:
        if c in df.columns:
            item_col = c
            break
    if item_col is None:
        raise ValueError(f"cannot find item_id column in {df.columns.tolist()}")

    corr_col = None
    for c in cand_corr:
        if c in df.columns:
            corr_col = c
            break
    if corr_col is None:
        raise ValueError(f"cannot find correct column in {df.columns.tolist()}")

    time_col = None
    for c in cand_time:
        if c in df.columns:
            time_col = c
            break
    if time_col is None:
        # 没时间列也行，我们造一个递增的
        time_col = None

    return item_col, corr_col, time_col

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="data/ednet_u200/KT1")
    ap.add_argument("--out",  default="analysis/ednet_flat_u200_v2.csv")
    args = ap.parse_args()

    rows = []
    for name in os.listdir(args.root):
        if not name.endswith(".csv"):
            continue
        fpath = os.path.join(args.root, name)
        uid = os.path.splitext(name)[0]   # u123456
        df = pd.read_csv(fpath)
        item_col, corr_col, time_col = detect_cols(df)
        if time_col is None:
            # 没有时间列，就用行号当时间
            df["_tmp_ts_"] = range(len(df))
            time_col = "_tmp_ts_"
        for _, r in df.iterrows():
            rows.append({
                "user_id": uid,
                "item_id": r[item_col],
                "correct": int(r[corr_col]),
                "timestamp": int(r[time_col]),
            })

    out_df = pd.DataFrame(rows)
    out_df = out_df.sort_values(["user_id", "timestamp"]).reset_index(drop=True)
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    out_df.to_csv(args.out, index=False)
    print(f"✅ wrote {args.out} rows={len(out_df)} from dir={args.root}")
