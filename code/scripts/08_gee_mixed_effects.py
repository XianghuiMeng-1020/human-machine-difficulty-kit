
import os, pandas as pd, numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
import statsmodels.api as sm
import statsmodels.formula.api as smf

Path("tables").mkdir(exist_ok=True); Path("figures").mkdir(exist_ok=True)

runs = pd.read_csv("data/race_runs.csv")
ord_map={"low":0.0,"middle":0.5,"high":1.0}
runs["is_correct"]=pd.to_numeric(runs["is_correct"], errors="coerce")
runs["human_diff_ord"]=runs["human_label"].map(ord_map)

# 读取认知标签（若无则跳过 GEE）
tagp="data/race/qid_cog_tag.csv"
if not os.path.exists(tagp):
    print("[SKIP] no data/race/qid_cog_tag.csv -> skip GEE mixed-like analysis.")
    raise SystemExit(0)
tags=pd.read_csv(tagp)
df=(runs.merge(tags, on="qid", how="left")
        .dropna(subset=["is_correct","human_diff_ord"])
        .assign(model_err=lambda d:1-d["is_correct"])
        .query("cog_tag.notnull() and cog_tag!=''")).copy()

if df.empty:
    print("[SKIP] df empty after merges.")
    raise SystemExit(0)

# GEE: binomial + logit，按 qid 聚类，带交互
# 公式： model_err ~ human_diff_ord * C(cog_tag) + C(model)
df["cog_tag"]=df["cog_tag"].astype("category")
df["model"]=df["model"].astype("category")
formula = "model_err ~ human_diff_ord * C(cog_tag) + C(model)"
fam = sm.families.Binomial()
ind = sm.cov_struct.Exchangeable()  # 题内相关

gee = smf.gee(formula, "qid", data=df, family=fam, cov_struct=ind)
fit = gee.fit()
tab = fit.summary().tables[1]
tab_df = pd.DataFrame(tab.data[1:], columns=tab.data[0])
tab_df.to_csv("tables/race_gee_item_cluster.csv", index=False)
print("[WROTE] tables/race_gee_item_cluster.csv")

# 边际效应：不同 cog_tag 下，human_diff_ord ∈ {0,0.5,1}
grid=[]
tags_sorted=sorted(df["cog_tag"].cat.categories.tolist())
for t in tags_sorted:
    for h in [0.0,0.5,1.0]:
        row={"cog_tag":t,"human_diff_ord":h}
        # 取模型众数作为对比参考
        row["model"]=df["model"].mode().iat[0]
        grid.append(row)
grid=pd.DataFrame(grid)

pred = fit.predict(grid)
grid["pred_err"]=pred

plt.figure(figsize=(6.6,4.4))
for t in tags_sorted:
    sub=grid[grid["cog_tag"]==t]
    plt.plot(sub["human_diff_ord"], sub["pred_err"], marker="o", label=t)
plt.xticks([0,0.5,1], ["low","middle","high"])
plt.ylim(0,1)
plt.xlabel("Human difficulty"); plt.ylabel("Predicted error rate")
plt.title("RACE: Predicted error vs difficulty by cognitive tag (GEE per-item cluster)")
plt.legend(frameon=False, bbox_to_anchor=(1.02,1), loc="upper left")
plt.tight_layout()
png="figures/race_gee_marginal_err_by_tag.png"
plt.savefig(png, dpi=160); plt.close()
print("[WROTE]", png)
print("[OK] 08_gee_mixed_effects 完成。")
