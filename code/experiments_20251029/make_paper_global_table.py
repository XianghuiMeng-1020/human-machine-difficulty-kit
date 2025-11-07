import os
import pandas as pd

SRC = "analysis/global/global_alignment_table.csv"
OUT = "paper_assets/RESULTS_GLOBAL_BASELINES.csv"

df = pd.read_csv(SRC)

# 1) 选我们真正想给 reviewer 看的那些
keep = []

for _, r in df.iterrows():
    ds = str(r["dataset"])
    var = str(r["variant"])
    mdl = str(r["model"])
    aln = r["alignment"]

    # 过滤掉空 alignment
    if pd.isna(aln):
        continue

    # ① 我们的方法主线
    if ds == "Eedi" and "MV-HMDA" in var:
        keep.append(r)
        continue
    if ds == "RACE" and "proxy-5runs" in var:
        keep.append(r)
        continue
    if ds.startswith("EdNet-KT1"):
        keep.append(r)
        continue

    # ② 三条 baseline 线
    if "BEA24" in var:
        keep.append(r)
        continue
    if "CL4KT" in var:
        keep.append(r)
        continue
    if "TempScaling" in var:
        keep.append(r)
        continue

paper = pd.DataFrame(keep).copy()

# 2) 排序：先我们，再 baseline
# 想要的顺序
order_ds = [
    "Eedi",
    "RACE",
    "EdNet-KT1",
    "mixed(eedi/race/syn)",
]
paper["__ds_order"] = paper["dataset"].apply(
    lambda x: next((i for i, d in enumerate(order_ds) if str(x).startswith(d)), 999)
)

# 大的在前
paper = paper.sort_values(["__ds_order", "alignment"], ascending=[True, False])

paper = paper.drop(columns=["__ds_order"])

os.makedirs("paper_assets", exist_ok=True)
paper.to_csv(OUT, index=False, encoding="utf-8")
print("✅ wrote", OUT, "rows=", len(paper))
