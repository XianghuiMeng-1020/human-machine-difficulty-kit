MIN_N = 2  # patched for demo overlap

import os, pandas as pd, numpy as np, matplotlib.pyplot as plt, seaborn as sns
from pathlib import Path

Path("tables").mkdir(exist_ok=True)
Path("figures").mkdir(exist_ok=True)

def _num(s): 
    return pd.to_numeric(s, errors="coerce")

def load_runs(p, name):
    if not os.path.exists(p): 
        return pd.DataFrame()
    df = pd.read_csv(p)
    keep = ["qid","model","is_correct","conf","human_label","diff_conf"]
    for k in keep:
        if k not in df.columns: 
            df[k] = np.nan
    df["dataset"] = name
    df["is_correct"] = _num(df["is_correct"])
    df["conf"] = _num(df["conf"])
    df["diff_conf"] = _num(df["diff_conf"])
    # ordinalize human_label if present
    lab = df["human_label"].astype(str).str.lower()
    map_ord = {"low":0.0,"middle":0.5,"mid":0.5,"med":0.5,"high":1.0,"1":1.0,"0":0.0}
    df["human_diff_ord"] = lab.map(map_ord)
    # prefer diff_conf if available; fallback to human_diff_ord
    df["human_diff"] = df["diff_conf"].where(df["diff_conf"].notna(), df["human_diff_ord"])
    return df[["dataset","qid","model","is_correct","conf","human_diff"]]

race = load_runs("data/race_runs.csv","race")
eedi = load_runs("data/eedi_runs.csv","eedi")
all_df = pd.concat([race, eedi], ignore_index=True)

# aggregate per qid,model
g = (all_df
     .groupby(["dataset","qid","model"], as_index=False)
     .agg(human_diff=("human_diff","mean"),
          is_correct_mean=("is_correct","mean"),
          mean_conf=("conf","mean"))
    )
g["model_err"] = 1 - g["is_correct_mean"]
g["one_minus_mean_conf"] = 1 - g["mean_conf"]

def safe_rho(a,b):
    a=_num(a); b=_num(b)
    a=a.dropna(); b=b.dropna()
    kk = a.index.intersection(b.index) if not a.index.equals(b.index) else a.index
    if len(kk)<3: return np.nan
    a,b = a.loc[kk], b.loc[kk]
    if a.nunique()<2 or b.nunique()<2: return np.nan
    return a.corr(b, method="spearman")

# per dataset x model correlations
rows=[]
for (ds,m), gg in g.groupby(["dataset","model"]):
    r_err = safe_rho(gg["human_diff"], gg["model_err"])
    r_1mc = safe_rho(gg["human_diff"], gg["one_minus_mean_conf"])
    n = int(len(gg))
    rows.append({"dataset":ds,"model":m,"n":n,"rho(human,err)":r_err,"rho(human,1mconf)":r_1mc})
per_ds = pd.DataFrame(rows)
per_ds.to_csv("tables/generalization_per_dataset.csv", index=False)

# wide to compare race vs eedi (generalization gap)
def _pivot(col):
    T = per_ds.pivot(index="model", columns="dataset", values=col)
    if "race" in T.columns and "eedi" in T.columns:
        T["gap"] = T["eedi"] - T["race"]
    return T

gap_err = _pivot("rho(human,err)")
gap_1mc = _pivot("rho(human,1mconf)")

gap_err.to_csv("tables/generalization_gap_err.csv")
gap_1mc.to_csv("tables/generalization_gap_1mconf.csv")

# pretty summary
summ = (per_ds
        .pivot_table(index="model", columns="dataset", values=["rho(human,err)","rho(human,1mconf)"])
        .copy())
for base, name in [("rho(human,err)","gap_err"), ("rho(human,1mconf)","gap_1mconf")]:
    if ("race" in summ[base]) and ("eedi" in summ[base]):
        summ[(base,"gap")] = summ[(base,"eedi")] - summ[(base,"race")]
summ.columns = [f"{a}|{b}" for a,b in summ.columns.to_flat_index()]
summ = summ.reset_index()
summ.to_csv("tables/generalization_gap_summary.csv", index=False)

# heatmap (race vs eedi) for both metrics
hm = per_ds.copy()
hm = hm.melt(id_vars=["model","dataset"], value_vars=["rho(human,err)","rho(human,1mconf)"],
             var_name="metric", value_name="rho")
hm["metric"] = hm["metric"].map({"rho(human,err)":"err", "rho(human,1mconf)":"1mconf"})

for met in ["err","1mconf"]:
    sub = hm[hm["metric"]==met].pivot(index="model", columns="dataset", values="rho")
    if sub.empty: 
        continue
    plt.figure(figsize=(5.0, 0.6+0.5*len(sub)))
    sns.heatmap(sub, annot=True, fmt=".2f", vmin=-1, vmax=1, cmap="vlag", cbar=True)
    plt.title(f"Generalization (Spearman ρ) — {met}")
    plt.tight_layout()
    outp = f"figures/generalization_heatmap_{met}.png"
    plt.savefig(outp, dpi=180); plt.close()
    print("[WROTE]", outp)

print("[WROTE] tables/generalization_per_dataset.csv")
print("[WROTE] tables/generalization_gap_err.csv")
print("[WROTE] tables/generalization_gap_1mconf.csv")
print("[WROTE] tables/generalization_gap_summary.csv")