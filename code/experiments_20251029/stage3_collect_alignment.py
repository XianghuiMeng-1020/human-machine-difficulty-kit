import csv, os

OUT = "paper_assets/mv-hmda/stage3_alignment_summary.csv"

rows = []

# 1) 原始（你那两个最早的 crosstab 其实都是 300 total 但 labeled 只有 5）
# 我们简单记录一下
rows.append({
    "dataset": "eedi",
    "model": "gpt4o",
    "mode": "raw_sparse",
    "alignment": 0.0067,
    "notes": "from original human_x_model_tau080.csv (sparse)"
})
rows.append({
    "dataset": "eedi",
    "model": "gpt4o-mini",
    "mode": "raw_sparse",
    "alignment": 0.0100,
    "notes": "from original human_x_model_tau080.csv (sparse)"
})

# 2) proxy 模式（刚刚你跑出来的）
rows.append({
    "dataset": "eedi",
    "model": "gpt4o",
    "mode": "proxy_dense",
    "alignment": 0.70,
    "notes": "from eedi_proxy_x_model_gpt4o_tau08.csv"
})
rows.append({
    "dataset": "eedi",
    "model": "gpt4o-mini",
    "mode": "proxy_dense",
    "alignment": 0.80,
    "notes": "from eedi_proxy_x_model_gpt4omini_tau08.csv"
})

os.makedirs(os.path.dirname(OUT), exist_ok=True)
with open(OUT, "w", newline="", encoding="utf-8") as f:
    hdr = ["dataset","model","mode","alignment","notes"]
    w = csv.DictWriter(f, fieldnames=hdr)
    w.writeheader()
    w.writerows(rows)

print("✅ wrote", OUT)
