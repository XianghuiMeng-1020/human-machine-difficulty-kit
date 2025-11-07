
import os, pandas as pd, numpy as np, statsmodels.api as sm
from patsy import dmatrices
from pathlib import Path
Path("tables").mkdir(exist_ok=True); Path("figures").mkdir(exist_ok=True)

runs = pd.read_csv("data/race_runs.csv")
if not os.path.exists("data/race/qid_cog_tag.csv"):
    print("[WARN] data/race/qid_cog_tag.csv missing -> skip"); raise SystemExit(0)
tags = pd.read_csv("data/race/qid_cog_tag.csv")

ord_map={"low":0.0,"middle":0.5,"high":1.0}
runs["is_correct"]=pd.to_numeric(runs["is_correct"], errors="coerce")
runs["human_diff_ord"]=runs["human_label"].map(ord_map)

df=(runs.merge(tags, on="qid", how="left")
        .dropna(subset=["is_correct","human_diff_ord"])
        .assign(model_err=lambda d:1-d["is_correct"])
        .query("cog_tag.notnull() and cog_tag!=''"))

if df.empty:
    print("[INFO] no rows with cog_tag; abort"); raise SystemExit(0)

rows=[]
for m in sorted(df["model"].dropna().unique()):
    sub=df[df["model"]==m].copy()
    if sub.empty: continue
    # 设定基类=按字母序第一个标签
    tags_sorted=sorted(sub["cog_tag"].dropna().unique())
    if not tags_sorted: continue
    baseline=tags_sorted[0]
    f"baseline={baseline}" # no-op to keep var used
    y, X = dmatrices("model_err ~ human_diff_ord + C(cog_tag, Treatment(reference='%s')) + human_diff_ord:C(cog_tag, Treatment(reference='%s'))" % (baseline, baseline),
                     sub, return_type="dataframe")
    model=sm.GLM(y, X, family=sm.families.Binomial())
    fit=model.fit(cov_type="cluster", cov_kwds={"groups": sub["qid"]})
    tab=fit.summary2().tables[1].reset_index().rename(columns={"index":"term"})
    tab.insert(0,"model",m)
    rows.append(tab)

if not rows:
    print("[INFO] nothing fitted"); raise SystemExit(0)

out=pd.concat(rows, ignore_index=True)
out.to_csv("tables/race_by_cogtag_glm_cluster_multi.csv", index=False)
print("[WROTE] tables/race_by_cogtag_glm_cluster_multi.csv  rows=", len(out))
