import argparse, os
import pandas as pd

"""
特征设计：
- prev_corr 砍掉！（这是刚刚最容易泄露的）
- hist_len_norm: 历史做过的题数 / 100
- hist_acc: 历史平均正确率
- hist_max_diff / hist_avg_diff: 用 item 表里的难度（-1→0）
- new_item_seen_global: 这个 next_item_id 在全局是不是热门题（用出现次数做个频率）
"""

def build_global_item_pop(steps_csv):
    df = pd.read_csv(steps_csv)
    cnt = df["item_id"].astype(str).value_counts()
    return cnt.to_dict(), len(df)

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--next_hard", default="baselines/cl4kt_diff/ednet_u200_nextstep_hard.csv")
    ap.add_argument("--steps",     default="baselines/cl4kt_diff/ednet_u200_steps.csv")
    ap.add_argument("--items",     default="baselines/cl4kt_diff/ednet_u200_items.csv")
    ap.add_argument("--out",       default="baselines/cl4kt_diff/ednet_u200_next_feats_hard.csv")
    args = ap.parse_args()

    next_df = pd.read_csv(args.next_hard)
    steps   = pd.read_csv(args.steps).sort_values(["user_id","step"])
    items   = pd.read_csv(args.items)
    diff_map = dict(zip(items["item_id"].astype(str), items["diff_int"].astype(int)))

    pop_map, pop_total = build_global_item_pop(args.steps)

    # 为了快速查用户历史
    steps_g = {uid: g for uid, g in steps.groupby("user_id")}

    feat_rows = []
    for row in next_df.itertuples(index=False):
        uid = row.user_id
        t_step = row.t_step
        y_next = row.y_next
        nxt_item = str(row.next_item_id)

        g = steps_g[uid]
        hist = g[g["step"] <= t_step].sort_values("step")

        if len(hist) > 0:
            hist_len_norm = min(len(hist), 100) / 100.0
            hist_acc = hist["correct"].mean()
            diffs = []
            for it in hist["item_id"].astype(str).tolist():
                diffs.append(diff_map.get(it, -1))
            if diffs:
                diffs2 = [d if d >= 0 else 0 for d in diffs]
                hist_max_diff = max(diffs2)
                hist_avg_diff = sum(diffs2) / len(diffs2)
            else:
                hist_max_diff = 0
                hist_avg_diff = 0.0
        else:
            hist_len_norm = 0.0
            hist_acc = 0.0
            hist_max_diff = 0
            hist_avg_diff = 0.0

        # 全局的 next_item 热度
        nxt_pop = pop_map.get(nxt_item, 0) / pop_total

        feat_rows.append({
            "user_id": uid,
            "t_step": t_step,
            "hist_len_norm": hist_len_norm,
            "hist_acc": hist_acc,
            "hist_max_diff": hist_max_diff,
            "hist_avg_diff": hist_avg_diff,
            "next_item_pop": nxt_pop,
            "label": y_next,
        })

    out_df = pd.DataFrame(feat_rows)
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    out_df.to_csv(args.out, index=False)
    print(f"✅ wrote {args.out} rows={len(out_df)}")
