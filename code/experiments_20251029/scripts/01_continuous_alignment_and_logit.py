import os, pandas as pd, numpy as np
from scipy import stats
import statsmodels.formula.api as smf
import matplotlib.pyplot as plt

os.makedirs("tables", exist_ok=True)
os.makedirs("figures", exist_ok=True)

def spearman_kendall(df, human_col, model_col, group_cols):
    rows = []
    for m, g in df.groupby(group_cols):
        x = g[human_col].values
        y = g[model_col].values
        if np.unique(x).size < 2 or np.unique(y).size < 2:
            rows.append((m if isinstance(m, str) else m[0], np.nan, np.nan, np.nan, np.nan, len(g)))
            continue
        rho, rp = stats.spearmanr(x, y)
        tau, tp = stats.kendalltau(x, y)
        rows.append((m if isinstance(m, str) else m[0], rho, rp, tau, tp, len(g)))
    return pd.DataFrame(rows, columns=[*group_cols, "spearman_rho", "spearman_p", "kendall_tau", "kendall_p", "n"])

def scatter_plot(df, human_col, model_col, title, outpath):
    plt.figure()
    plt.scatter(df[human_col], df[model_col], s=12, alpha=0.6)
    if df[human_col].nunique() > 1:
        b = np.polyfit(df[human_col], df[model_col], 1)
        xg = np.linspace(df[human_col].min(), df[human_col].max(), 100)
        plt.plot(xg, b[0]*xg + b[1])
    plt.xlabel(human_col); plt.ylabel(model_col); plt.title(title)
    plt.tight_layout(); plt.savefig(outpath, dpi=180); plt.close()

# ---------- RACE ----------
if os.path.exists("data/race_runs.csv"):
    race = pd.read_csv("data/race_runs.csv")
    race["human_diff"] = race["human_label"].map({"middle":0, "high":1}).astype(float)
    grp = race.groupby(["qid","model"], as_index=False).agg(
        acc=("is_correct","mean"), mean_conf=("conf","mean"), std_conf=("conf","std")
    )
    grp["model_err"] = 1.0 - grp["acc"]
    grp["one_minus_mean_conf"] = 1.0 - grp["mean_conf"]

    c1 = spearman_kendall(grp, "human_diff", "model_err", ["model"])
    c2 = spearman_kendall(grp, "human_diff", "one_minus_mean_conf", ["model"])
    c3 = spearman_kendall(grp, "human_diff", "std_conf", ["model"])
    c1.merge(c2, on="model", suffixes=("_err", "_1mconf")).merge(c3, on="model", suffixes=("", "_std")).to_csv(
        "tables/race_continuous_alignment_correlations.csv", index=False
    )
    for m in grp["model"].unique():
        g = grp[grp["model"]==m]
        scatter_plot(g, "human_diff", "model_err", f"RACE {m}: human_diff vs model_err", f"figures/race_{m}_human_vs_model_err.png")
        scatter_plot(g, "human_diff", "one_minus_mean_conf", f"RACE {m}: human_diff vs 1-mean_conf", f"figures/race_{m}_human_vs_1mconf.png")
        scatter_plot(g, "human_diff", "std_conf", f"RACE {m}: human_diff vs std_conf", f"figures/race_{m}_human_vs_stdconf.png")

    m1 = smf.glm("is_correct ~ human_diff + C(model) + C(qid)", data=race, family=__import__('statsmodels.api').families.Binomial()).fit(
        cov_type="cluster", cov_kwds={"groups": race["qid"]}
    )
    m2 = smf.glm("is_correct ~ human_diff * C(model) + C(qid)", data=race, family=__import__('statsmodels.api').families.Binomial()).fit(
        cov_type="cluster", cov_kwds={"groups": race["qid"]}
    )
    m1.summary2().tables[1].reset_index().rename(columns={"index":"term"}).to_csv("tables/race_logit_fe_main.csv", index=False)
    m2.summary2().tables[1].reset_index().rename(columns={"index":"term"}).to_csv("tables/race_logit_fe_interaction.csv", index=False)
else:
    print("WARN: data/race_runs.csv not found; skipping RACE.")

# ---------- Eedi ----------
if os.path.exists("data/eedi_runs.csv"):
    eedi = pd.read_csv("data/eedi_runs.csv")
    grp = eedi.groupby(["qid","model"], as_index=False).agg(
        acc=("is_correct","mean"), mean_conf=("conf","mean"), diff_conf=("diff_conf","mean")
    )
    grp["model_err"] = 1.0 - grp["acc"]
    grp["one_minus_mean_conf"] = 1.0 - grp["mean_conf"]

    c1 = spearman_kendall(grp, "diff_conf", "model_err", ["model"])
    c2 = spearman_kendall(grp, "diff_conf", "one_minus_mean_conf", ["model"])
    c1.merge(c2, on="model", suffixes=("_err","_1mconf")).to_csv("tables/eedi_continuous_alignment_correlations.csv", index=False)

    for m in grp["model"].unique():
        g = grp[grp["model"]==m]
        scatter_plot(g, "diff_conf", "model_err", f"Eedi {m}: diff_conf vs model_err", f"figures/eedi_{m}_diff_vs_model_err.png")
        scatter_plot(g, "diff_conf", "one_minus_mean_conf", f"Eedi {m}: diff_conf vs 1-mean_conf", f"figures/eedi_{m}_diff_vs_1mconf.png")

    eedi["diff_conf_std"] = (eedi["diff_conf"] - eedi["diff_conf"].mean()) / (eedi["diff_conf"].std() + 1e-8)
    m1 = smf.glm("is_correct ~ diff_conf_std + C(model) + C(qid)", data=eedi, family=__import__('statsmodels.api').families.Binomial()).fit(
        cov_type="cluster", cov_kwds={"groups": eedi["qid"]}
    )
    m2 = smf.glm("is_correct ~ diff_conf_std * C(model) + C(qid)", data=eedi, family=__import__('statsmodels.api').families.Binomial()).fit(
        cov_type="cluster", cov_kwds={"groups": eedi["qid"]}
    )
    m1.summary2().tables[1].reset_index().rename(columns={"index":"term"}).to_csv("tables/eedi_logit_fe_main.csv", index=False)
    m2.summary2().tables[1].reset_index().rename(columns={"index":"term"}).to_csv("tables/eedi_logit_fe_interaction.csv", index=False)
else:
    print("WARN: data/eedi_runs.csv not found; skipping Eedi.")

print("Done. Wrote tables/ and figures/.")
