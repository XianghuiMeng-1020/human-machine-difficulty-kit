import argparse, os
import pandas as pd
import numpy as np

def build_feats(steps_csv, items_csv, max_len=100):
    steps = pd.read_csv(steps_csv)
    items = pd.read_csv(items_csv)
    diff_map = dict(zip(items["item_id"].astype(str), items["diff_int"].astype(int)))

    rows = []
    # 按 user 聚合，再按时间顺序滑
    for uid, g in steps.groupby("user_id"):
        g = g.sort_values("step")
        hist_items = []
        hist_corrects = []
        hist_diffs = []

        for _, r in g.iterrows():
            it = str(r["item_id"])
            co = int(r["correct"])
            # 当前这一步要预测的标签：就是这一步是否做对
            label = co

            # 用历史构造特征（注意：不含当前这题）
            if hist_items:
                prev_corr = hist_corrects[-1]
                seq_len = min(len(hist_items), max_len) / max_len
                # 历史平均正确率
                hist_acc = sum(hist_corrects) / len(hist_corrects)
                # 历史难度（用 max 表示有没有做过“很难”的题）
                if hist_diffs:
                    max_diff = max(hist_diffs)
                    avg_diff = sum(hist_diffs) / len(hist_diffs)
                else:
                    max_diff = 0
                    avg_diff = 0.0
            else:
                prev_corr = 0
                seq_len = 0.0
                hist_acc = 0.0
                max_diff = 0
                avg_diff = 0.0

            # 当前题目的难度不能直接用，我们只能看看“这个题在全局是不是没见过”
            cur_diff = diff_map.get(it, -1)
            # 我们做一个 very weak 的特征：这个题有没有出现在历史中
            seen_cur_item = 1.0 if it in hist_items else 0.0

            feat = {
                "user_id": uid,
                "item_id": it,
                "label": label,
                "prev_corr": prev_corr,
                "seq_len": seq_len,
                "hist_acc": hist_acc,
                "hist_max_diff": max_diff,   # 取值 -1/2... 但我们上面转成了 0 起步
                "hist_avg_diff": avg_diff,
                "seen_cur_item": seen_cur_item,
                # 这里特意不放 it_hash，防止泄露
            }
            rows.append(feat)

            # 把当前信息再塞进历史，给下一步用
            hist_items.append(it)
            hist_corrects.append(co)
            # diff, 这里保留原值（-1 也行）
            hist_diffs.append(diff_map.get(it, -1))

    return pd.DataFrame(rows)

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", default="baselines/cl4kt_diff/ednet_u200_steps.csv")
    ap.add_argument("--items", default="baselines/cl4kt_diff/ednet_u200_items.csv")
    ap.add_argument("--out",   default="baselines/cl4kt_diff/ednet_u200_feats_strict.csv")
    args = ap.parse_args()

    df = build_feats(args.steps, args.items)
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    df.to_csv(args.out, index=False)
    print(f"✅ wrote {args.out} rows={len(df)}")
