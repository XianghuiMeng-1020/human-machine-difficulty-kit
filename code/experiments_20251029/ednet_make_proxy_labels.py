import csv, argparse, os
import numpy as np
from collections import defaultdict

if __name__=="__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--logs", default="analysis/ednet_kt1_flat_small.csv")
    ap.add_argument("--out",  default="analysis/ednet_proxy_labels.csv")
    ap.add_argument("--min_cnt", type=int, default=5)
    args = ap.parse_args()

    per_item = defaultdict(list)
    with open(args.logs, newline="", encoding="utf-8") as f:
        r = csv.DictReader(f)
        for row in r:
            per_item[row["item_id"]].append(int(row["correct"]))

    rows = []
    usable_errs = []
    for iid, corr_list in per_item.items():
        cnt = len(corr_list)
        err = 1 - np.mean(corr_list)
        rows.append({"item_id": iid, "err_rate": err, "count": cnt})
        if cnt >= args.min_cnt:
            usable_errs.append(err)

    # 尝试分位
    use_quantile = False
    if len(usable_errs) >= 30:
        q1, q2 = np.quantile(usable_errs, [0.33, 0.66])
        # 如果分位有效（不是都一样），就用
        if not (abs(q1 - q2) < 1e-6):
            use_quantile = True
    if not use_quantile:
        # 退回固定阈值：<0.4 易, <0.7 中, 否则难
        q1, q2 = 0.4, 0.7

    for r in rows:
        e, cnt = r["err_rate"], r["count"]
        if cnt < args.min_cnt:
            r["H_proxy"] = "低曝光"
        else:
            if e < q1:
                r["H_proxy"] = "简单H_proxy"
            elif e < q2:
                r["H_proxy"] = "中等H_proxy"
            else:
                r["H_proxy"] = "困难H_proxy"

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w", newline="", encoding="utf-8") as f:
        hdr = ["item_id","err_rate","count","H_proxy"]
        w = csv.DictWriter(f, fieldnames=hdr); w.writeheader(); w.writerows(rows)

    # 统计一下
    from collections import Counter
    dist = Counter([r["H_proxy"] for r in rows])
    print("✅ wrote", args.out, "items:", len(rows))
    print("use_quantile:", use_quantile, "q1=%.3f q2=%.3f" % (q1, q2))
    print("dist:", dict(dist))
