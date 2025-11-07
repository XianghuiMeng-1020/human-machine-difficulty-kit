import csv, math, os, argparse

def sigmoid(x):
    return 1 / (1 + math.exp(-x))

def calibrate_p(p, alpha):
    # 防止 0/1
    eps = 1e-6
    p = min(max(p, eps), 1 - eps)
    logit = math.log(p / (1 - p))
    logit_scaled = alpha * logit
    return sigmoid(logit_scaled)

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--scores", default="data/eedi_gpt4o_300x1/scores.csv")
    ap.add_argument("--proxy",  default="analysis/eedi_proxy_labels.csv")
    ap.add_argument("--taus",   default="0.8,0.9")
    ap.add_argument("--alphas", default="0.8,1.0,1.2")
    ap.add_argument("--out",    default="paper_assets/mv-hmda/stage3_calibration_ablation_eedi.csv")
    args = ap.parse_args()

    taus   = [float(x) for x in args.taus.split(",")]
    alphas = [float(x) for x in args.alphas.split(",")]

    # 1) 读 proxy
    proxy = {}
    with open(args.proxy, newline="", encoding="utf-8") as f:
        r = csv.DictReader(f)
        for row in r:
            proxy[row["question_id"]] = row["H_proxy"]

    # 2) 读 scores
    records = []
    with open(args.scores, newline="", encoding="utf-8") as f:
        r = csv.DictReader(f)
        for row in r:
            records.append({
                "qid": row["question_id"],
                "correct": int(row["correct"]),
                "p": float(row["p_chosen"]),
            })

    out_rows = []
    for tau in taus:
        for alpha in alphas:
            aligned = 0
            total   = 0
            for rec in records:
                qid = rec["qid"]
                if qid not in proxy:
                    continue
                p_cal = calibrate_p(rec["p"], alpha)
                # 模型端打标
                if rec["correct"] == 1 and p_cal >= tau:
                    mlabel = "简单M"
                elif rec["correct"] == 0 and p_cal >= tau:
                    mlabel = "困难M"
                else:
                    mlabel = "中等M"
                # proxy 映射
                # 简单H_proxy, 中等H_proxy, 困难H_proxy
                hlabel = proxy[qid]
                mapM = {"简单M": "简单H_proxy",
                        "中等M": "中等H_proxy",
                        "困难M": "困难H_proxy"}
                if mlabel in mapM:
                    total += 1
                    if mapM[mlabel] == hlabel:
                        aligned += 1
            align_ratio = aligned / total if total else 0.0
            out_rows.append({
                "dataset": "eedi",
                "model": "gpt4o",
                "tau": tau,
                "alpha": alpha,
                "n": total,
                "alignment": round(align_ratio, 4),
            })

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w", newline="", encoding="utf-8") as f:
        hdr = ["dataset","model","tau","alpha","n","alignment"]
        w = csv.DictWriter(f, fieldnames=hdr)
        w.writeheader()
        w.writerows(out_rows)

    print("✅ wrote", args.out)
    for r in out_rows:
        print(r)
