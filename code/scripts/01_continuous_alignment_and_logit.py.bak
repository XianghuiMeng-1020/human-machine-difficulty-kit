import os, pandas as pd, numpy as np
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import stats
import statsmodels.api as sm


def safe_write(df: pd.DataFrame, path: str):
    """
    写 CSV：把数值列里的 NaN 作为空字符串导出，但不在原 DataFrame 的浮点列里直接
    赋值 ""（那会触发 FutureWarning）。这里先复制，再把需要导出的列转成 object，
    仅在副本中把缺失值替换为 ""，最后写盘。
    """
    import numpy as np
    df_out = df.copy()

    for c in df_out.columns:
        s = df_out[c]
        if pd.api.types.is_numeric_dtype(s):
            # 数值列：先转 object，再只把缺失位置替换为 ""，其余保持为数值
            obj = s.astype(object)
            # 注意：不能直接 obj[s.isna()] = "" 写在 float 列上，否则会有 FutureWarning
            mask = s.isna()
            if mask.any():
                obj[mask] = ""
            df_out[c] = obj
        else:
            # 非数值列：统一把缺失显示为空串
            df_out[c] = s.where(s.notna(), "")

    df_out.to_csv(path, index=False)

def scatter(df, x, y, title, out):
    g=df[[x,y]].dropna()
    if len(g)==0: return
    plt.figure()
    plt.scatter(g[x], g[y], s=10)
    plt.title(title)
    plt.xlabel(x)
    plt.ylabel(y)
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out, bbox_inches="tight")
    plt.close()

def safe_corr(x, y):
    if x is None or y is None: return "", "", 0, "", "", 0
    a=pd.Series(x); b=pd.Series(y)
    m=a.notna() & b.notna()
    a=a[m]; b=b[m]
    if len(a)==0: return "", "", 0, "", "", 0
    if a.nunique()<2 or b.nunique()<2: return "", "", 0, "", "", 0
    sr=stats.spearmanr(a, b)
    kr=stats.kendalltau(a, b)
    s_rho=float(sr.correlation) if np.isfinite(sr.correlation) else ""
    s_p=float(sr.pvalue) if np.isfinite(sr.pvalue) else ""
    k_tau=float(kr.correlation) if np.isfinite(kr.correlation) else ""
    k_p=float(kr.pvalue) if np.isfinite(kr.pvalue) else ""
    n=len(a)
    return s_rho, s_p, n, k_tau, k_p, n

def glm_tables(df_xy, xcol, ycol, main_path, inter_path):
    X = sm.add_constant(df_xy[[xcol]].astype(float), has_constant="add")
    y = df_xy[ycol].astype(float)
    try:
        model = sm.GLM(y, X, family=sm.families.Binomial())
        res = model.fit()
        t = res.summary2().tables[1].reset_index().rename(columns={"index":"term"})
        safe_write(t, main_path)
        safe_write(t, inter_path)
    except Exception:
        ols = sm.OLS(y, X, missing="drop").fit()
        ci = ols.conf_int()
        out = pd.DataFrame({
            "term": ["Intercept", xcol],
            "Coef.": [ols.params.get("const", np.nan), ols.params.get(xcol, np.nan)],
            "Std.Err.": [ols.bse.get("const", np.nan), ols.bse.get(xcol, np.nan)],
            "z": [ols.tvalues.get("const", np.nan), ols.tvalues.get(xcol, np.nan)],
            "P>|z|": [ols.pvalues.get("const", np.nan), ols.pvalues.get(xcol, np.nan)],
            "[0.025": [ci.loc["const",0] if "const" in ci.index else np.nan,
                       ci.loc[xcol,0] if xcol in ci.index else np.nan],
            "0.975]": [ci.loc["const",1] if "const" in ci.index else np.nan,
                       ci.loc[xcol,1] if xcol in ci.index else np.nan],
        })
        safe_write(out, main_path)
        safe_write(out, inter_path)

if os.path.exists("data/race_runs.csv"):
    race=pd.read_csv("data/race_runs.csv")
    if len(race)==0 or "human_label" not in race.columns:
        pass
    else:
        race["human_diff"]=(race["human_label"].astype(str).str.lower()=="high").astype(float)
        vc=race.groupby("model")["qid"].nunique()
        keep_models=vc.index.tolist()
        race=race[race["model"].isin(keep_models)].copy()
        grp=race.groupby(["qid","model"],as_index=False).agg(
            is_correct_mean=("is_correct", lambda s: pd.to_numeric(s, errors="coerce").mean()),
            mean_conf=("conf", lambda s: pd.to_numeric(s, errors="coerce").mean()),
            std_conf=("conf", lambda s: pd.to_numeric(s, errors="coerce").std(ddof=0)),
            human_diff=("human_diff","first")
        )
        grp["model_err"]=1-grp["is_correct_mean"]
        grp["one_minus_mean_conf"]=1-grp["mean_conf"]
        rows=[]
        for m in sorted(grp["model"].unique()):
            g=grp[grp["model"]==m].copy()
            s1_rho,s1_p,n1,k1_tau,k1_p,_=safe_corr(g["human_diff"], g["model_err"])
            s2_rho,s2_p,n2,k2_tau,k2_p,_=safe_corr(g["human_diff"], g["one_minus_mean_conf"])
            s3_rho,s3_p,n3,k3_tau,k3_p,_=safe_corr(g["human_diff"], g["std_conf"])
            rows.append({
                "model":m,
                "spearman_rho_err":s1_rho,"spearman_p_err":s1_p,"kendall_tau_err":k1_tau,"kendall_p_err":k1_p,"n_err":n1,
                "spearman_rho_1mconf":s2_rho,"spearman_p_1mconf":s2_p,"kendall_tau_1mconf":k2_tau,"kendall_p_1mconf":k2_p,"n_1mconf":n2,
                "spearman_rho":s3_rho,"spearman_p":s3_p,"kendall_tau":k3_tau,"kendall_p":k3_p,"n":n3
            })
            scatter(g,"human_diff","model_err",f"RACE {m}: human_diff vs model_err",f"figures/race_{m}_human_vs_model_err.png")
            if n2>0:
                scatter(g,"human_diff","one_minus_mean_conf",f"RACE {m}: human_diff vs 1-mean_conf",f"figures/race_{m}_human_vs_1mconf.png")
            if n3>0:
                scatter(g,"human_diff","std_conf",f"RACE {m}: human_diff vs std_conf",f"figures/race_{m}_human_vs_stdconf.png")
        out=pd.DataFrame(rows)
        # --- sanitize to avoid NaN in CSV when n==0 ---
        corr_cols = [
            "spearman_rho_1mconf","spearman_p_1mconf",
            "kendall_tau_1mconf","kendall_p_1mconf",
            "spearman_rho","spearman_p","kendall_tau","kendall_p"
        ]
        count_cols = ["n_1mconf","n"]
        for c in count_cols:
            if c in out.columns:
                out[c] = out[c].fillna(0).astype(int)
        for c in corr_cols:
            if c in out.columns:
                out[c] = out[c].astype(object)
                out[c] = out[c].where(pd.notna(out[c]), "")

        safe_write(out,"tables/race_continuous_alignment_correlations.csv")
        mlist=sorted(grp["model"].unique())
        if len(mlist)>0:
            mm=mlist[0]
            gg=grp[grp["model"]==mm].dropna(subset=["human_diff","model_err"])
            if len(gg)>0 and gg["human_diff"].nunique()>1 and gg["model_err"].nunique()>1:
                glm_tables(gg, "human_diff", "model_err", "tables/race_logit_agg_main.csv", "tables/race_logit_agg_interaction.csv")
            else:
                empty=pd.DataFrame({"term":["Intercept","human_diff"],"Coef.":[np.nan,np.nan],"Std.Err.":[np.nan,np.nan],"z":[np.nan,np.nan],"P>|z|":[np.nan,np.nan],"[0.025":[np.nan,np.nan],"0.975]":[np.nan,np.nan]})
                safe_write(empty,"tables/race_logit_agg_main.csv")
                safe_write(empty,"tables/race_logit_agg_interaction.csv")
else:
    pass
