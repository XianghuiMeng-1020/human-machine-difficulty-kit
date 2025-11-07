import argparse, os
import pandas as pd

"""
从一个 very big 的 EdNet 平铺表里做 per-item 的人类难度 proxy
输入表必须至少有：
- item_id
- correct   (0/1, 我们刚刚在 02_apply_pseudo_contents 里打过了)
我们会分块统计：
- cnt_i     = 这个 item 出现了几次
- err_i     = 1 - (sum(correct)/cnt)
然后按阈值分桶：
- cnt < min_cnt         → 低曝光
- else & err >= hi_thr  → 困难H_proxy
- else & err <= lo_thr  → 简单H_proxy   (可选，看你要不要)
- 其它                   → 中等H_proxy
"""

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--logs", default="analysis/ednet_flat_with_correct.csv",
                    help="95M 的那个大 csv")
    ap.add_argument("--out", default="analysis/ednet_proxy_labels_full.csv")
    ap.add_argument("--chunksize", type=int, default=500_000)
    ap.add_argument("--min_cnt", type=int, default=5,
                    help="少于这个次数的题归到 低曝光")
    ap.add_argument("--lo_thr", type=float, default=0.4,
                    help="err <= 0.4 认为是简单 (可选)")
    ap.add_argument("--hi_thr", type=float, default=0.7,
                    help="err >= 0.7 认为是困难")
    args = ap.parse_args()

    # 累计表：item_id -> [cnt, correct_sum]
    agg = {}

    total = 0
    for chunk in pd.read_csv(args.logs, chunksize=args.chunksize):
        # 保守起见都转成 str / int
        if "item_id" not in chunk.columns or "correct" not in chunk.columns:
            raise ValueError(f"chunk missing columns: {chunk.columns.tolist()}")

        for row in chunk.itertuples(index=False):
            it = str(row.item_id)
            corr = int(row.correct) if row.correct == row.correct else 0
            if it not in agg:
                agg[it] = [0, 0]
            agg[it][0] += 1
            agg[it][1] += corr

        total += len(chunk)
        print(f"[pass] processed {total} rows ...")

    # 汇总成 df
    rows = []
    for it, (cnt, corr_sum) in agg.items():
        acc = corr_sum / cnt if cnt > 0 else 0.0
        err = 1.0 - acc
        # 先默认一列
        rows.append({
            "item_id": it,
            "cnt": cnt,
            "acc": acc,
            "err": err
        })
    df = pd.DataFrame(rows)

    # 打标签
    labels = []
    for r in df.itertuples(index=False):
        if r.cnt < args.min_cnt:
            labels.append("低曝光")
        else:
            if r.err >= args.hi_thr:
                labels.append("困难H_proxy")
            elif r.err <= args.lo_thr:
                labels.append("简单H_proxy")
            else:
                labels.append("中等H_proxy")
    df["H_proxy"] = labels

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    df.to_csv(args.out, index=False)
    print(f"✅ wrote {args.out} items: {len(df)}")

    # 简单分布看看
    dist = df["H_proxy"].value_counts().to_dict()
    print("dist:", dist)

if __name__ == "__main__":
    main()
