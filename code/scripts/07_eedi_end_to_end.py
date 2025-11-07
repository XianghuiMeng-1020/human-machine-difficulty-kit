
import os, pandas as pd, numpy as np, glob
from pathlib import Path

Path("data/eedi").mkdir(parents=True, exist_ok=True)
Path("experiments_20251029/results/eedi").mkdir(parents=True, exist_ok=True)

def _safe_read_csv(p, **kw):
    try:
        return pd.read_csv(p, **kw)
    except Exception:
        return pd.DataFrame()

# 1) 原始题库模板
raw_p = "data/eedi_raw_items.csv"
if not os.path.exists(raw_p):
    pd.DataFrame({
        "qid": [], "stem": [], "choice_A": [], "choice_B": [],
        "choice_C": [], "choice_D": [], "gold_choice": []
    }).to_csv(raw_p, index=False)
    print(f"[TEMPLATE] {raw_p} 已创建（空），如需完整 EEDI 请填充后重跑。")

# 2) 人类难度映射模板
diff_map_p = "data/eedi/human_diff_map.csv"
if not os.path.exists(diff_map_p):
    raw = _safe_read_csv(raw_p)
    qids = (raw.get("qid") if "qid" in raw.columns else pd.Series([], dtype=str)).astype(str)
    pd.DataFrame({"qid": qids, "human_label": "", "diff_conf": ""}).to_csv(diff_map_p, index=False)
    print(f"[TEMPLATE] {diff_map_p} 已创建（空标签）。")

# 3) 汇聚 per_question -> data/eedi_runs.csv
rows=[]
for perq in sorted(glob.glob("experiments_20251029/results/eedi/*/round1/per_question.csv")):
    try:
        df = pd.read_csv(perq)
    except Exception:
        continue
    if df.empty: 
        continue
    model = perq.split("/")[3]
    qid  = df["qid"].astype(str) if "qid" in df.columns else pd.Series([None]*len(df))
    ic   = pd.to_numeric(df.get("is_correct"), errors="coerce")
    conf = pd.to_numeric(df.get("conf"), errors="coerce")
    rows.append(pd.DataFrame({"qid": qid, "model": model, "is_correct": ic, "conf": conf}))

runs = pd.concat(rows, ignore_index=True) if rows else pd.DataFrame(columns=["qid","model","is_correct","conf"])
runs.to_csv("data/eedi_runs.csv", index=False)
print(f"[WROTE] data/eedi_runs.csv rows={len(runs)} models={sorted(runs['model'].dropna().unique()) if len(runs) else []}")

# 4) 合并 diff_map；确保输出一定包含 human_label/diff_conf 两列
out = _safe_read_csv("data/eedi_runs.csv")
mp  = _safe_read_csv(diff_map_p)
if not out.empty and not mp.empty:
    mp["qid"] = mp["qid"].astype(str).str.strip()
    out["qid"] = out["qid"].astype(str).str.strip()
    out = out.merge(mp[["qid","human_label","diff_conf"]], on="qid", how="left")
# 强制存在列
for col in ("human_label","diff_conf"):
    if col not in out.columns:
        out[col] = np.nan
out.to_csv("data/eedi_runs.csv", index=False)

# 覆盖率（健壮计算）
if len(out)==0:
    cov = 0.0
else:
    dc = pd.to_numeric(out["diff_conf"], errors="coerce")
    cov = float(pd.Series(dc).notna().mean())
print("[EEDI] merged human_diff_map -> data/eedi_runs.csv  coverage(diff_conf)=", cov)

# 5) 指标（与 RACE 同制式），为空也写空表，不报错
from sklearn.metrics import roc_auc_score

def ece_brier(df, bins=15):
    d = df.dropna(subset=["conf","is_correct"]).copy()
    if d.empty: return "", ""
    d["bin"]=np.clip((d["conf"]*bins).astype(int),0,bins-1)
    g = d.groupby("bin").agg(acc=("is_correct","mean"), conf=("conf","mean"), n=("conf","size"))
    if g["n"].sum()==0: return "", ""
    e = (g["n"]*abs(g["acc"]-g["conf"])).sum()/g["n"].sum()
    b = ((d["conf"]-d["is_correct"])**2).mean()
    return f"{e:.3f}", f"{b:.3f}"

auc_rows=[]; cal_rows=[]; rc_lines=[]
if not out.empty and "model" in out.columns:
    out["is_correct"]=pd.to_numeric(out["is_correct"], errors="coerce")
    out["conf"]=pd.to_numeric(out["conf"], errors="coerce")
    for m in sorted(out["model"].dropna().unique()):
        d = out[(out["model"]==m)]
        dd = d.dropna(subset=["conf","is_correct"])
        if dd.empty or dd["is_correct"].nunique()<2:
            auc_rows.append({"dataset":"eedi","model":m,"ROC_AUC":""})
        else:
            auc_rows.append({"dataset":"eedi","model":m,"ROC_AUC":f"{roc_auc_score(dd['is_correct'], dd['conf']):.3f}"})
        E,B = ece_brier(d)
        cal_rows.append({"dataset":"eedi","model":m,"ECE(15bins)":E,"Brier":B})

pd.DataFrame(auc_rows).to_csv("tables/eedi_conf_discrimination_auc.csv", index=False)
pd.DataFrame(cal_rows).to_csv("tables/eedi_calibration_summary.csv", index=False)
print("[OK] 07_eedi_end_to_end 完成。")
