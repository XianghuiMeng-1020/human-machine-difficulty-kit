import csv, argparse, os, numpy as np
from collections import defaultdict, Counter

def label_covaware(log_csv, out_csv, low_thr=5, high_thr=20):
    per_item = defaultdict(list)
    with open(log_csv, newline="", encoding="utf-8") as f:
        r = csv.DictReader(f)
        for row in r:
            iid = row["item_id"]
            corr = int(row["correct"])
            per_item[iid].append(corr)

    items = []
    hi_errs = []
    for iid, corr_list in per_item.items():
        cnt = len(corr_list)
        err = 1 - np.mean(corr_list)
        items.append({"item_id": iid, "err_rate": err, "count": cnt})
        if cnt >= high_thr:
            hi_errs.append(err)

    if len(hi_errs) >= 50:
        hq1, hq2 = np.quantile(hi_errs, [0.33, 0.66])
        if abs(hq1 - hq2) < 1e-6:
            hq1, hq2 = 0.4, 0.7
    else:
        hq1, hq2 = 0.4, 0.7

    for rec in items:
        e, cnt = rec["err_rate"], rec["count"]
        if cnt < low_thr:
            rec["H_proxy"] = "低曝光"
        elif cnt < high_thr:
            if e < 0.4:   rec["H_proxy"] = "简单H_proxy"
            elif e < 0.7: rec["H_proxy"] = "中等H_proxy"
            else:         rec["H_proxy"] = "困难H_proxy"
        else:
            if e < hq1:    rec["H_proxy"] = "简单H_proxy"
            elif e < hq2:  rec["H_proxy"] = "中等H_proxy"
            else:          rec["H_proxy"] = "困难H_proxy"

    os.makedirs(os.path.dirname(out_csv), exist_ok=True)
    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        hdr = ["item_id","err_rate","count","H_proxy"]
        w = csv.DictWriter(f, fieldnames=hdr); w.writeheader(); w.writerows(items)

    dist = Counter([x["H_proxy"] for x in items])
    print(f"✅ wrote {out_csv} items: {len(items)} dist: {dict(dist)} thresholds: {(hq1, hq2)}")

if __name__=="__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--logs", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--low_thr", type=int, default=5)
    ap.add_argument("--high_thr", type=int, default=20)
    args = ap.parse_args()
    label_covaware(args.logs, args.out, args.low_thr, args.high_thr)
