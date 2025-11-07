import os, pandas as pd, numpy as np
from scipy import stats
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

os.makedirs("tables", exist_ok=True)
os.makedirs("figures", exist_ok=True)

df = pd.read_csv("data/race_runs.csv")
# 期望至少有列: qid, model, is_correct, conf, human_label（middle/high）
# 有些版本 human_label 可能叫 difficulty / label；兜底一下
if "human_label" not in df.columns:
    for k in ["difficulty","label","human_difficulty","race_label"]:
        if k in df.columns:
            df = df.rename(columns={k:"human_label"})
            break
if "human_label" not in df.columns:
    raise SystemExit("race_runs.csv 缺少 human_label（或 difficulty/label），无法按人类难度分组。")

df["model_err"] = 1 - df["is_correct"].astype(int)
df["conf"] = pd.to_numeric(df["conf"], errors="coerce")

# 将 middle/high 转成 0/1
labmap = {"middle":0, "Mid":0, "MID":0, "MIDDLE":0,
          "high":1,   "High":1, "HIGH":1}
df["human_bin"] = df["human_label"].map(labmap)
if df["human_bin"].isna().any():
    # 其他取值一律当作 1（更难）
    df["human_bin"] = df["human_bin"].fillna(1).astype(int)

g = df.groupby(["qid","human_label"], as_index=False).agg(
    acc=("is_correct","mean"),
    mean_conf=("conf","mean"),
    std_conf=("conf","std")
)
g["model_err"] = 1 - g["acc"]

# 人类难度（0/1） vs 模型误差、vs (1-mean_conf)
tmp = g.copy()
tmp["human_bin"] = tmp["human_label"].map(labmap).fillna(1).astype(int)
tmp["one_minus_conf"] = 1 - tmp["mean_conf"]

def safe_corr(x, y):
    x = x.to_numpy(); y = y.to_numpy()
    ok = np.isfinite(x) & np.isfinite(y)
    if ok.sum() < 3: return np.nan, np.nan, np.nan, np.nan
    rho,  p1 = stats.spearmanr(x[ok], y[ok])
    tau,  p2 = stats.kendalltau(x[ok], y[ok])
    return rho, p1, tau, p2

rho_e, p_e, tau_e, p2_e = safe_corr(tmp["human_bin"], tmp["model_err"])
rho_c, p_c, tau_c, p2_c = safe_corr(tmp["human_bin"], tmp["one_minus_conf"])

corr_tab = pd.DataFrame([{
    "model": "stage3",
    "spearman_rho_err": rho_e, "spearman_p_err": p_e,
    "kendall_tau_err": tau_e,  "kendall_p_err": p2_e,
    "spearman_rho_1mconf": rho_c, "spearman_p_1mconf": p_c,
    "kendall_tau_1mconf": tau_c,  "kendall_p_1mconf": p2_c,
    "n": len(tmp)
}])
corr_tab.to_csv("tables/race_alignment_correlations_stage3.csv", index=False)

by_human = g.groupby("human_label", as_index=False).agg(
    n=("qid","count"),
    acc=("acc","mean"),
    mean_conf=("mean_conf","mean"),
    std_conf=("mean_conf","std")
).sort_values("human_label")
by_human.to_csv("tables/race_stage3_by_human.csv", index=False)

def tag_by_tau(row, tau=0.80):
    c = row["mean_conf"]
    a = row["acc"]
    if not np.isfinite(c): return "mediumM"
    if c >= tau and a >= 0.5: return "simpleM"
    if c >= tau and a <  0.5: return "hardM"
    return "mediumM"
g["m_tag"] = g.apply(tag_by_tau, axis=1)
cross = pd.crosstab(g["human_label"], g["m_tag"]).reset_index()
cross.to_csv("tables/race_stage3_humanXmodel_tau080.csv", index=False)

gc = g.dropna(subset=["mean_conf"]).sort_values("mean_conf", ascending=False).reset_index(drop=True)
gc["covered"] = (np.arange(len(gc))+1)/len(gc)
gc["cum_acc"] = (gc["acc"].cumsum())/(np.arange(len(gc))+1)
gc[["covered","cum_acc"]].to_csv("tables/race_stage3_risk_coverage.csv", index=False)

plt.figure()
plt.plot(gc["covered"], gc["cum_acc"])
plt.xlabel("Coverage (by descending confidence)")
plt.ylabel("Cumulative accuracy")
plt.title("RACE stage3 — Risk-Coverage")
plt.tight_layout(); plt.savefig("figures/race_stage3_risk_coverage.png", dpi=180); plt.close()

def scatter(x, y, xl, yl, outpng):
    plt.figure()
    plt.scatter(x, y, s=12, alpha=0.6)
    plt.xlabel(xl); plt.ylabel(yl)
    plt.tight_layout(); plt.savefig(outpng, dpi=180); plt.close()

scatter(tmp["human_bin"], tmp["model_err"], "Human difficulty (high=1)", "Model error", "figures/race_stage3_human_vs_model_err_scatter.png")
scatter(tmp["human_bin"], 1-tmp["mean_conf"], "Human difficulty (high=1)", "1 - mean confidence", "figures/race_stage3_human_vs_1mconf_scatter.png")
# boxplot 也补一张
plt.figure()
tmp.boxplot(column="model_err", by="human_label")
plt.suptitle(""); plt.title("Model error by human difficulty")
plt.tight_layout(); plt.savefig("figures/race_stage3_err_by_human_box.png", dpi=180); plt.close()

print("Done 04. Wrote tables/:",
      "race_alignment_correlations_stage3.csv,",
      "race_stage3_by_human.csv,",
      "race_stage3_humanXmodel_tau080.csv,",
      "race_stage3_risk_coverage.csv")
