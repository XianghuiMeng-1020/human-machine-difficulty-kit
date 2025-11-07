import argparse, os
import pandas as pd

"""
把 per-step 的表(含 user_id, step, item_id, correct)
变成 “用 step=t 的信息预测 step=t+1 的正确性” 的表。

输出字段：
  user_id, t_item_id, t_correct, t_step,
  y_next   # 要预测的标签 = 下一题是否做对
"""

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", default="baselines/cl4kt_diff/ednet_u200_steps.csv")
    ap.add_argument("--out",   default="baselines/cl4kt_diff/ednet_u200_nextstep.csv")
    args = ap.parse_args()

    df = pd.read_csv(args.steps)
    df = df.sort_values(["user_id", "step"])

    rows = []
    for uid, g in df.groupby("user_id"):
        g = g.sort_values("step").reset_index(drop=True)
        for i in range(len(g) - 1):  # 最后一条没有下一个
            cur = g.iloc[i]
            nxt = g.iloc[i + 1]
            rows.append({
                "user_id": uid,
                "t_step":  int(cur["step"]),
                "t_item_id": str(cur["item_id"]),
                "t_correct": int(cur["correct"]),
                # 这个才是我们要预测的
                "y_next": int(nxt["correct"]),
            })
    out_df = pd.DataFrame(rows)
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    out_df.to_csv(args.out, index=False)
    print(f"✅ wrote {args.out} rows={len(out_df)}")
