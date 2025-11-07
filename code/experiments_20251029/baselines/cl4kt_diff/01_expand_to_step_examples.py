import argparse, os
import pandas as pd

"""
输入：
  baselines/cl4kt_diff/ednet_u200_sessions.csv
    user_id,items,corrects

输出：
  baselines/cl4kt_diff/ednet_u200_steps.csv
    user_id,step,item_id,correct,label
  其中 step 从 1 开始，表示这个用户的第 step 条交互
"""

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--sessions", default="baselines/cl4kt_diff/ednet_u200_sessions.csv")
    ap.add_argument("--out", default="baselines/cl4kt_diff/ednet_u200_steps.csv")
    args = ap.parse_args()

    sess = pd.read_csv(args.sessions)
    rows = []
    for row in sess.itertuples(index=False):
        items = str(row.items).split()
        cors  = str(row.corrects).split()
        for i, (it, co) in enumerate(zip(items, cors), start=1):
            rows.append({
                "user_id": row.user_id,
                "step": i,
                "item_id": it,
                "correct": int(float(co)),
            })

    out_df = pd.DataFrame(rows)
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    out_df.to_csv(args.out, index=False)
    print(f"✅ wrote {args.out} rows={len(out_df)} from {args.sessions}")
