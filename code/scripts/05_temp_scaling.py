
import os, pandas as pd, numpy as np, matplotlib.pyplot as plt
from pathlib import Path
Path("tables").mkdir(exist_ok=True); Path("figures").mkdir(exist_ok=True)

def logit(p):
    p=np.clip(p,1e-6,1-1e-6); return np.log(p/(1-p))

def sigmoid(z): return 1/(1+np.exp(-z))

def nll(p,y):
    p=np.clip(p,1e-6,1-1e-6); y=np.asarray(y)
    return - (y*np.log(p)+(1-y)*np.log(1-p)).mean()

def ece_brier(df, bins=15):
    d=df.dropna(subset=["conf","is_correct"]).copy()
    if d.empty: return "", ""
    d["bin"]=np.clip((d["conf"]*bins).astype(int),0,bins-1)
    g=d.groupby("bin").agg(acc=("is_correct","mean"), conf=("conf","mean"), n=("conf","size"))
    E=(g["n"]*abs(g["acc"]-g["conf"])).sum()/g["n"].sum()
    B=((d["conf"]-d["is_correct"])**2).mean()
    return f"{E:.3f}", f"{B:.3f}"

runs=pd.read_csv("data/race_runs.csv")
runs["conf"]=pd.to_numeric(runs.get("conf"), errors="coerce")
runs["is_correct"]=pd.to_numeric(runs.get("is_correct"), errors="coerce")

rows=[]
for m in sorted(runs["model"].dropna().unique()):
    d=runs[(runs["model"]==m) & runs["conf"].notna() & runs["is_correct"].notna()].copy()
    if d.empty: 
        rows.append({"model":m,"T*":"","ECE_raw":"","Brier_raw":"","ECE_scaled":"","Brier_scaled":""})
        continue
    # 简单cal/val划分：qid哈希奇偶
    mask = d["qid"].astype(str).map(lambda s: (sum(map(ord,s))%2)==0)
    cal, val = d[mask].copy(), d[~mask].copy()
    if cal.empty or val.empty:
        cal, val = d.iloc[:len(d)//2].copy(), d.iloc[len(d)//2:].copy()
    # 网格搜 T
    base_logit = logit(cal["conf"].values)
    ys = cal["is_correct"].values
    Ts=np.linspace(0.5,5.0,91)
    bestT, bestLoss = None, 1e9
    for T in Ts:
        loss = nll(sigmoid(base_logit/T), ys)
        if loss<bestLoss: bestLoss, bestT = loss, T
    # 应用
    val_scaled = val.copy()
    val_scaled["conf"] = sigmoid(logit(val_scaled["conf"].values)/bestT)
    e_raw, b_raw = ece_brier(val)
    e_sc,  b_sc  = ece_brier(val_scaled)
    rows.append({"model":m,"T*":f"{bestT:.2f}","ECE_raw":e_raw,"Brier_raw":b_raw,"ECE_scaled":e_sc,"Brier_scaled":b_sc})

    # 可靠性曲线（原/缩放）
    for tag, df_ in [("raw", val), ("scaled", val_scaled)]:
        kk=df_.copy(); kk["bin"]=np.clip((kk["conf"]*15).astype(int),0,14)
        g=kk.groupby("bin").agg(acc=("is_correct","mean"), conf=("conf","mean")).reset_index()
        if len(g):
            plt.figure(figsize=(4.8,4.0)); plt.plot(g["conf"], g["acc"], marker="o"); plt.plot([0,1],[0,1],"--")
            plt.xlabel("Confidence"); plt.ylabel("Accuracy"); plt.title(f"RACE {m}: Reliability ({tag})"); plt.tight_layout()
            png=f"figures/race_{m}_reliability_{tag}.png"; plt.savefig(png, dpi=160); plt.close(); print("[WROTE]", png)

out=pd.DataFrame(rows)
out.to_csv("tables/race_calibration_temp_scaling.csv", index=False)
print("[WROTE] tables/race_calibration_temp_scaling.csv")
print(out.to_string(index=False))
