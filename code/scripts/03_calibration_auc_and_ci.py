
import os, numpy as np, pandas as pd
from pathlib import Path
from sklearn.metrics import roc_auc_score
import matplotlib.pyplot as plt
from scipy import stats as st

Path("tables").mkdir(exist_ok=True); Path("figures").mkdir(exist_ok=True)

runs = pd.read_csv("data/race_runs.csv")
# 基础清洗
runs["conf"] = pd.to_numeric(runs.get("conf"), errors="coerce")
runs["is_correct"] = pd.to_numeric(runs.get("is_correct"), errors="coerce")

# human difficulty：优先 diff_conf；否则用 label -> {low:0, middle:0.5, high:1}
if "diff_conf" in runs.columns:
    runs["human_diff_ord"] = pd.to_numeric(runs["diff_conf"], errors="coerce")
else:
    ord_map = {"low":0.0, "middle":0.5, "high":1.0}
    runs["human_diff_ord"] = runs.get("human_label", "").map(ord_map)

def conf_is_usable(d: pd.DataFrame) -> bool:
    c = d["conf"].dropna()
    return (len(c) > 0) and (c.nunique() > 1)

def fully_separable(d: pd.DataFrame) -> bool:
    d = d.dropna(subset=["conf","is_correct"]).copy()
    if d.empty or d["is_correct"].nunique()<2: return False
    pos = d.loc[d["is_correct"]==1, "conf"]
    neg = d.loc[d["is_correct"]==0, "conf"]
    return (len(pos)>0 and len(neg)>0 and pos.min() > neg.max())

def looks_leaky(d: pd.DataFrame) -> bool:
    d = d.dropna(subset=["conf","is_correct"]).copy()
    if d.empty: return False
    c = np.round(d["conf"], 3); y = d["is_correct"]
    rule1 = set(c.unique()) <= {0.25, 0.75}
    rule2 = (np.round((c-0.25)/0.5, 3) - y).abs().sum() == 0
    return bool(rule1 or rule2)

def ece_brier(d: pd.DataFrame, bins: int = 15):
    dd = d.dropna(subset=["conf","is_correct"]).copy()
    if dd.empty: return "", ""
    dd["bin"] = np.clip((dd["conf"]*bins).astype(int), 0, bins-1)
    g = dd.groupby("bin").agg(acc=("is_correct","mean"), conf=("conf","mean"), n=("conf","size"))
    if g["n"].sum() == 0: return "", ""
    E = (g["n"] * (g["acc"] - g["conf"]).abs()).sum() / g["n"].sum()
    B = ((dd["conf"] - dd["is_correct"])**2).mean()
    return f"{E:.3f}", f"{B:.3f}"

def boot_ci(x, y, stat="spearman", B=2000, seed=0):
    a = pd.to_numeric(x, errors="coerce")
    b = pd.to_numeric(y, errors="coerce")
    ok = a.notna() & b.notna()
    a, b = a[ok].to_numpy(), b[ok].to_numpy()
    n = len(a)
    if n < 3 or len(np.unique(a)) < 2 or len(np.unique(b)) < 2:
        return "", "", 0
    rng = np.random.default_rng(seed)
    vals = []
    for _ in range(B):
        idx = rng.integers(0, n, n)
        if stat == "spearman":
            vals.append(st.spearmanr(a[idx], b[idx]).correlation)
        else:
            vals.append(st.kendalltau(a[idx], b[idx]).correlation)
    lo, hi = np.nanpercentile(vals, [2.5, 97.5])
    base = st.spearmanr(a, b).correlation if stat=="spearman" else st.kendalltau(a, b).correlation
    return f"{base:.3f}", f"[{lo:.3f}, {hi:.3f}]", n

# ---------- AUC / ECE / Brier ----------
auc_rows, cal_rows, rc_rows = [], [], []
for m in sorted(runs["model"].dropna().unique()):
    dm = runs[runs["model"]==m]
    usable = conf_is_usable(dm) and (dm["is_correct"].nunique()>=2) and (not fully_separable(dm)) and (not looks_leaky(dm))
    if usable:
        d = dm.dropna(subset=["conf","is_correct"])
        auc = roc_auc_score(d["is_correct"], d["conf"])
        auc_rows.append({"model":m, "ROC_AUC":f"{auc:.3f}"})
        E,B = ece_brier(dm, bins=15)
        cal_rows.append({"model":m, "ECE(15bins)":E, "Brier":B})
        # 风险-覆盖曲线数据
        kk = dm.dropna(subset=["conf","is_correct"]).sort_values("conf", ascending=False).copy()
        kk["rank"] = np.arange(1, len(kk)+1)
        kk["covered"] = kk["rank"] / len(kk)
        kk["cum_acc"] = kk["is_correct"].expanding().mean()
        kk["mean_conf"] = kk["conf"].expanding().mean()
        tmp = kk[["covered","cum_acc","mean_conf"]].copy()
        tmp.insert(0, "model", m)
        rc_rows.extend(tmp.to_dict("records"))
    else:
        auc_rows.append({"model":m, "ROC_AUC":""})
        cal_rows.append({"model":m, "ECE(15bins)":"", "Brier":""})

pd.DataFrame(auc_rows).to_csv("tables/race_conf_discrimination_auc.csv", index=False)
pd.DataFrame(cal_rows).to_csv("tables/race_calibration_summary.csv", index=False)
pd.DataFrame(rc_rows).to_csv("tables/race_risk_coverage.csv", index=False)

# === OVERRIDE: risk-coverage from runs ===
import pandas as _pd, numpy as _np, os as _os, pathlib as _pl
_rc_runs="data/race_runs.csv"
if _os.path.exists(_rc_runs):
    _R=_pd.read_csv(_rc_runs)
    if set(["model","1mconf","acc"]).issubset(_R.columns):
        _rows=[]
        for _m,_df in _R.groupby("model"):
            _df=_df.sort_values("1mconf", ascending=False).reset_index(drop=True)
            _n=len(_df)
            _df["cum_correct"]=_df["acc"].cumsum()
            _df["covered"]=(_np.arange(_n)+1)/_n
            _df["cum_acc"]=_df["cum_correct"]/(_np.arange(_n)+1)
            _df["mean_conf"]=_df["1mconf"].expanding().mean()
            _rows.append(_df[["model","covered","cum_acc","mean_conf"]])
        _out=_pd.concat(_rows, ignore_index=True)
        _pl.Path("tables").mkdir(parents=True, exist_ok=True)
        _out.to_csv("tables/race_risk_coverage.csv", index=False)
        print("[OVERRIDE] tables/race_risk_coverage.csv written from runs:", _out.shape)


# ---------- 相关性的 Bootstrap CI（修正：用 human_diff_ord vs model_err / 1−mean_conf） ----------
agg = runs.groupby(["qid","model"], as_index=False).agg(
    is_correct_mean=("is_correct", "mean"),
    mean_conf=("conf", "mean"),
    human_diff_ord=("human_diff_ord", "mean"),  # 每题应为常数
)
agg["model_err"] = 1 - agg["is_correct_mean"]
agg["one_minus_mean_conf"] = 1 - agg["mean_conf"]

rows=[]
for m in sorted(agg["model"].dropna().unique()):
    g = agg[agg["model"]==m].copy()
    # 保证有人类难度
    if g["human_diff_ord"].notna().sum() == 0:
        rows.append({"model":m,"metric":"err","n":0,"spearman_rho":"","spearman_rho_CI95":"","kendall_tau":"","kendall_tau_CI95":""})
        rows.append({"model":m,"metric":"1mconf","n":0,"spearman_rho":"","spearman_rho_CI95":"","kendall_tau":"","kendall_tau_CI95":""})
        continue
    # err
    rho, rho_ci, n = boot_ci(g["human_diff_ord"], g["model_err"], "spearman")
    tau, tau_ci, n2 = boot_ci(g["human_diff_ord"], g["model_err"], "kendall")
    rows.append({"model":m, "metric":"err", "n":max(n,n2),
                 "spearman_rho":rho, "spearman_rho_CI95":rho_ci,
                 "kendall_tau":tau, "kendall_tau_CI95":tau_ci})
    # 1mconf
    rho, rho_ci, n = boot_ci(g["human_diff_ord"], g["one_minus_mean_conf"], "spearman")
    tau, tau_ci, n2 = boot_ci(g["human_diff_ord"], g["one_minus_mean_conf"], "kendall")
    rows.append({"model":m, "metric":"1mconf", "n":max(n,n2),
                 "spearman_rho":rho, "spearman_rho_CI95":rho_ci,
                 "kendall_tau":tau, "kendall_tau_CI95":tau_ci})

ci = pd.DataFrame(rows)
for c in ["spearman_rho","spearman_rho_CI95","kendall_tau","kendall_tau_CI95"]:
    ci[c] = ci[c].astype(object).where(ci[c].notna(), "")
ci.to_csv("tables/race_continuous_alignment_correlations_CI.csv", index=False)

# 预览
def peek(p, n=8):
    if os.path.exists(p):
        import pandas as pd
        print(f"\n== {p} ==")
        try: print(pd.read_csv(p).head(n).to_string(index=False))
        except Exception as e: print("[read fail]", e)

peek("tables/race_conf_discrimination_auc.csv")
peek("tables/race_calibration_summary.csv")
peek("tables/race_risk_coverage.csv")
peek("tables/race_continuous_alignment_correlations_CI.csv")
print("\n[OK] 03_calibration_auc_and_ci 修正完成。")
