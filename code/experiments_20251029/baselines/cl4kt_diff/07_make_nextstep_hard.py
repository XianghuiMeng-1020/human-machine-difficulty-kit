import argparse, os
import pandas as pd

"""
hard 版约束：
1) 用户按时间排
2) 我们只取 (t -> t+1) 中，下一个 item 是“这个用户第一次见到的 item”
3) 我们还会在最后过滤掉：这个用户所有 label 都是 1 或 0 的情况
"""

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", default="baselines/cl4kt_diff/ednet_u200_steps.csv")
    ap.add_argument("--out",   default="baselines/cl4kt_diff/ednet_u200_nextstep_hard.csv")
    args = ap.parse_args()

    steps = pd.read_csv(args.steps).sort_values(["user_id","step"])
    rows = []
    for uid, g in steps.groupby("user_id"):
        g = g.sort_values("step").reset_index(drop=True)
        seen_items = set()
        for i in range(len(g) - 1):
            cur = g.iloc[i]
            nxt = g.iloc[i+1]
            nxt_item = str(nxt["item_id"])
            # 只要“下一个题是第一次出现的”
            if nxt_item in seen_items:
                # 已经见过，就跳过这条
                pass
            else:
                rows.append({
                    "user_id": uid,
                    "t_step": int(cur["step"]),
                    "t_item_id": str(cur["item_id"]),
                    "t_correct": int(cur["correct"]),
                    "y_next": int(nxt["correct"]),
                    "next_item_id": nxt_item,
                })
            # 更新 seen
            seen_items.add(str(cur["item_id"]))
        # 最后一条不做
    hard_df = pd.DataFrame(rows)

    # 过滤掉 label 单一的用户
    keep_uids = []
    for uid, g in hard_df.groupby("user_id"):
        ys = g["y_next"].unique().tolist()
        if len(ys) >= 2:   # 至少有 0 和 1
            keep_uids.append(uid)
    hard_df = hard_df[hard_df["user_id"].isin(keep_uids)].reset_index(drop=True)

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    hard_df.to_csv(args.out, index=False)
    print(f"✅ wrote {args.out} rows={len(hard_df)} after user-filter={len(keep_uids)} users")
