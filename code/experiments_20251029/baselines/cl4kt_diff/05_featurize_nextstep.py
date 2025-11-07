import argparse, os
import pandas as pd

def build_feats(nextstep_csv, steps_csv, items_csv, max_len=100):
    next_df = pd.read_csv(nextstep_csv)
    steps = pd.read_csv(steps_csv).sort_values(["user_id","step"])
    items = pd.read_csv(items_csv)
    diff_map = dict(zip(items["item_id"].astype(str), items["diff_int"].astype(int)))

    # 为了快速查历史，先按 user 分组
    steps_g = {uid: g for uid, g in steps.groupby("user_id")}

    rows = []
    for row in next_df.itertuples(index=False):
        uid = row.user_id
        t_step = row.t_step
        y_next = row.y_next

        g = steps_g[uid]
        hist = g[g["step"] <= t_step].sort_values("step")

        # 构造历史特征
        if len(hist) > 0:
            prev_corr = int(hist.iloc[-1]["correct"])
            seq_len = min(len(hist), max_len) / max_len
            hist_acc = hist["correct"].mean()
            # 历史题目的难度
            diffs = []
            for it in hist["item_id"].astype(str).tolist():
                diffs.append(diff_map.get(it, -1))
            if diffs:
                # 把 -1 当作 0
                diffs2 = [d if d >= 0 else 0 for d in diffs]
                hist_max_diff = max(diffs2)
                hist_avg_diff = sum(diffs2) / len(diffs2)
            else:
                hist_max_diff = 0
                hist_avg_diff = 0.0
        else:
            prev_corr = 0
            seq_len = 0.0
            hist_acc = 0.0
            hist_max_diff = 0
            hist_avg_diff = 0.0

        rows.append({
            "user_id": uid,
            "t_step": t_step,
            "prev_corr": prev_corr,
            "seq_len": seq_len,
            "hist_acc": hist_acc,
            "hist_max_diff": hist_max_diff,
            "hist_avg_diff": hist_avg_diff,
            "label": y_next,
        })

    return pd.DataFrame(rows)

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--nextstep", default="baselines/cl4kt_diff/ednet_u200_nextstep.csv")
    ap.add_argument("--steps",    default="baselines/cl4kt_diff/ednet_u200_steps.csv")
    ap.add_argument("--items",    default="baselines/cl4kt_diff/ednet_u200_items.csv")
    ap.add_argument("--out",      default="baselines/cl4kt_diff/ednet_u200_next_feats.csv")
    args = ap.parse_args()

    df = build_feats(args.nextstep, args.steps, args.items)
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    df.to_csv(args.out, index=False)
    print(f"✅ wrote {args.out} rows={len(df)}")
