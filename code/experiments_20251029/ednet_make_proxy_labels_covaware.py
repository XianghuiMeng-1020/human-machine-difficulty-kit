import csv, argparse, os
import numpy as np
from collections import defaultdict, Counter

if __name__=="__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--logs", default="analysis/ednet_kt1_flat_small.csv")
    ap.add_argument("--out",  default="analysis/ednet_proxy_labels_covaware.csv")
    # 档位阈值
    ap.add_argument("--low_thr",  type=int, default=5)   # <5 → 低曝光
    ap.add_argument("--high_thr", type=int, default=20)  # ≥20 → 高曝光，用分位
    args = ap.parse_args()

    per_item = defaultdict(list)
    with open(args.logs, newline="", encoding="utf-8") as f:
        r = csv.DictReader(f)
        for row in r:
            per_item[row["item_id"]].append(int(row["correct"]))

    items = []
    hi_errs = []   # 高曝光的 err
    mid_errs = []  # 中曝光的 err
    for iid, corr_list in per_item.items():
        cnt = len(corr_list)
        err = 1 - np.mean(corr_list)
        rec = {"item_id": iid, "err_rate": err, "count": cnt}
        items.append(rec)
        if cnt >= args.high_thr:
            hi_errs.append(err)
        elif cnt >= args.low_thr:
            mid_errs.append(err)

    # 高曝光：用分位；如果还是塌了就退回 0.4/0.7
    if len(hi_errs) >= 30:
        hq1, hq2 = np.quantile(hi_errs, [0.33, 0.66])
        if abs(hq1 - hq2) < 1e-6:
            hq1, hq2 = 0.4, 0.7
    else:
        hq1, hq2 = 0.4, 0.7

    # 中曝光：直接 0.4/0.7
    mq1, mq2 = 0.4, 0.7

    # 打标签
    for rec in items:
        e, cnt = rec["err_rate"], rec["count"]
        if cnt < args.low_thr:
            rec["H_proxy"] = "低曝光"
        elif cnt < args.high_thr:
            # 中曝光档
            if e < mq1:
                rec["H_proxy"] = "简单H_proxy"
            elif e < mq2:
                rec["H_proxy"] = "中等H_proxy"
            else:
                rec["H_proxy"] = "困难H_proxy"
        else:
            # 高曝光档
            if e < hq1:
                rec["H_proxy"] = "简单H_proxy"
            elif e < hq2:
                rec["H_proxy"] = "中等H_proxy"
            else:
                rec["H_proxy"] = "困难H_proxy"

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w", newline="", encoding="utf-8") as f:
        hdr = ["item_id","err_rate","count","H_proxy"]
        w = csv.DictWriter(f, fieldnames=hdr); w.writeheader(); w.writerows(items)

    # 统计
    dist = Counter([r["H_proxy"] for r in items])
    print("✅ wrote", args.out, "items:", len(items))
    print("dist:", dict(dist))
    print("high-exp thresholds:", hq1, hq2)
