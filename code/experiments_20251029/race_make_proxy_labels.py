import csv, os
from collections import defaultdict

CANON = "paper_assets/mv-hmda_race/stage1_canonical_race.csv"
OUT   = "paper_assets/mv-hmda_race/stage2_proxy_labels_race.csv"

# 读 canonical
by_qm = defaultdict(list)  # (qid, model) -> list of rows
with open(CANON, newline="", encoding="utf-8") as f:
    r = csv.DictReader(f)
    for row in r:
        qid = row["qid"]; m = row["model"]
        by_qm[(qid, m)].append(row)

# 先按 (qid, model) 聚合
per_q_model = defaultdict(dict)  # qid -> model -> dict(acc, mean_p)
for (qid, m), rows in by_qm.items():
    accs = [int(x["correct"]) for x in rows]
    ps   = [float(x["p_chosen"]) for x in rows]
    acc_mean = sum(accs)/len(accs)
    p_mean   = sum(ps)/len(ps)
    per_q_model[qid][m] = {
        "acc": acc_mean,
        "mean_p": p_mean,
    }

rows_out = []
for qid, md in per_q_model.items():
    # 我们只看这三个模型里出现的
    models = list(md.keys())
    accs   = [md[m]["acc"] for m in models]
    ps     = [md[m]["mean_p"] for m in models]

    # 规则
    if len(models) >= 2 and all(a >= 0.8 for a in accs) and all(p >= 0.8 for p in ps):
        h = "简单H_proxy"
    elif len(models) >= 2 and all(a <= 0.4 for a in accs) and any(p >= 0.8 for p in ps):
        h = "困难H_proxy"
    else:
        h = "中等H_proxy"

    rows_out.append({
        "question_id": qid,
        "H_proxy": h,
    })

os.makedirs(os.path.dirname(OUT), exist_ok=True)
with open(OUT, "w", newline="", encoding="utf-8") as f:
    hdr = ["question_id","H_proxy"]
    w = csv.DictWriter(f, fieldnames=hdr)
    w.writeheader()
    w.writerows(rows_out)

print("✅ wrote", OUT, "n=", len(rows_out))
