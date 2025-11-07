import os
import pandas as pd
import matplotlib.pyplot as plt

csv_path = "analysis/global/alignment_runs_all.csv"
df = pd.read_csv(csv_path)

# 统一大小写
df["dataset"] = df["dataset"].str.strip()
df["dataset_norm"] = df["dataset"].str.lower()

# 给每一行生成一个 label，方便画图
def make_label(row):
    ds = row["dataset"]
    var = row["variant"]
    model = row["model"]
    if isinstance(model, str) and model != "-" and model != "":
        return f"{ds}-{model}-{var}"
    else:
        return f"{ds}-{var}"

df["label"] = df.apply(make_label, axis=1)

# 排个序：先按 dataset，再按 alignment
df = df.sort_values(by=["dataset_norm", "alignment"], ascending=[True, False]).reset_index(drop=True)

os.makedirs("figs/global", exist_ok=True)

# ===== 图1：所有行的条形图 =====
plt.figure(figsize=(10, 4))
plt.bar(range(len(df)), df["alignment"])
plt.xticks(range(len(df)), df["label"], rotation=70, ha="right", fontsize=7)
plt.ylim(0, 1.05)
plt.ylabel("alignment")
plt.title("Alignment across datasets / variants / models")
plt.tight_layout()
out1 = "figs/global/alignment_all_rows.png"
plt.savefig(out1, dpi=300)
print("✅ saved", out1)

# ===== 图2：dataset 内部对比（base vs joint-head）=====

# 我们只挑能“成对”的：Eedi 和 RACE
pairs = df[(df["dataset"].isin(["Eedi", "RACE"]))].copy()
# 透视成宽表：index=(dataset,model), columns=variant
pivot = pairs.pivot_table(index=["dataset","model"], columns="variant", values="alignment")
# 画出来
plt.figure(figsize=(6, 4))
x_labels = []
x = []
base_vals = []
head_vals = []
i = 0
for (ds, m), row in pivot.iterrows():
    x_labels.append(f"{ds}-{m}")
    x.append(i)
    base_vals.append(row.get("base", float("nan")))
    head_vals.append(row.get("joint-head", float("nan")))
    i += 1

width = 0.35
plt.bar([xi - width/2 for xi in x], base_vals, width=width, label="base")
plt.bar([xi + width/2 for xi in x], head_vals, width=width, label="joint-head")
plt.xticks(x, x_labels, rotation=30, ha="right")
plt.ylim(0, 1.05)
plt.ylabel("alignment")
plt.title("Base vs joint-head (Eedi / RACE)")
plt.legend()
plt.tight_layout()
out2 = "figs/global/alignment_base_vs_joint.png"
plt.savefig(out2, dpi=300)
print("✅ saved", out2)
