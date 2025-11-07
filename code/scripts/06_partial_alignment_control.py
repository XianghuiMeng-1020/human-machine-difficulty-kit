
import os, pandas as pd, numpy as np, statsmodels.api as sm

def text_len(s):
    if not isinstance(s,str): return 0
    return len(s.split())

def word_overlap(q,c):
    if not isinstance(q,str) or not isinstance(c,str): return 0.0
    qa=set(q.lower().split()); ca=set(c.lower().split())
    if not qa or not ca: return 0.0
    return len(qa & ca) / len(qa | ca)

# ---------- load ----------
runs_path = "data/race_runs.csv"
raw_path  = "data/race_raw_items.csv"
if not os.path.exists(runs_path):
    print(f"[ERR] {runs_path} not found"); raise SystemExit(1)
runs = pd.read_csv(runs_path)

# is_correct / conf
runs["is_correct"]=pd.to_numeric(runs.get("is_correct"), errors="coerce")
runs["conf"]=pd.to_numeric(runs.get("conf"), errors="coerce")

# human_diff (ordinal 0/0.5/1); tolerate missing human_label
ord_map={"low":0.0,"middle":0.5,"high":1.0}
if "human_diff" not in runs.columns:
    if "human_label" in runs.columns:
        runs["human_diff"]=runs["human_label"].map(ord_map)
    else:
        # if neither exists, abort gracefully
        print("[WARN] no human_label/human_diff in runs; skip partial correlation")
        pd.DataFrame(columns=["model","partial_rho_err","partial_rho_1mconf","n"]).to_csv(
            "tables/race_partial_alignment_summary.csv", index=False)
        raise SystemExit(0)

# ---------- item features ----------
if not os.path.exists(raw_path):
    print("[WARN] data/race_raw_items.csv missing -> skip")
    pd.DataFrame(columns=["model","partial_rho_err","partial_rho_1mconf","n"]).to_csv(
        "tables/race_partial_alignment_summary.csv", index=False)
    raise SystemExit(0)

raw = pd.read_csv(raw_path)
raw["len_context"]=raw.get("context","").map(text_len)
raw["len_question"]=raw.get("question","").map(text_len)
raw["overlap"]=raw.apply(lambda r: word_overlap(r.get("question",""), r.get("context","")), axis=1)
feat=raw[["qid","len_context","len_question","overlap"]]

# ---------- per-qid per-model aggregation (CRITICAL: carry human_diff) ----------
grp=(runs.groupby(["qid","model"], as_index=False)
        .agg(is_correct_mean=("is_correct","mean"),
             mean_conf=("conf","mean"),
             human_diff=("human_diff","max"))  # constant per qid; any of max/mean/first works
        .merge(feat, on="qid", how="left"))

grp["model_err"]=1-grp["is_correct_mean"]

# drop rows with missing human_diff
grp=grp.dropna(subset=["human_diff"])

rows=[]
for m in sorted(grp["model"].dropna().unique()):
    g=grp[grp["model"]==m].dropna(subset=["human_diff","model_err"])
    if len(g)<10:
        rows.append({"model":m,"partial_rho_err":"","partial_rho_1mconf":"","n":len(g)})
        continue

    # rank-transform (Spearman style) then residualize vs controls
    controls=["len_context","len_question","overlap"]
    for col in ["human_diff","model_err","mean_conf"]:
        g[col+"_r"]=g[col].rank(method="average", pct=True)

    X = sm.add_constant(g[[c for c in controls if c in g.columns]].fillna(0.0))
    r_h = sm.OLS(g["human_diff_r"],      X).fit().resid
    r_e = sm.OLS(g["model_err_r"],       X).fit().resid
    r_c = sm.OLS((1 - g["mean_conf_r"]), X).fit().resid  # 1 - conf vs difficulty

    def corr(a,b):
        a=pd.Series(a); b=pd.Series(b)
        a=a-a.mean(); b=b-b.mean()
        denom = np.sqrt((a*a).sum()*(b*b).sum())
        return float((a*b).sum()/denom) if denom>0 else np.nan

    rho_err = corr(r_h, r_e)
    rho_1mc = corr(r_h, r_c)
    rows.append({"model":m,
                 "partial_rho_err": ("" if pd.isna(rho_err) else f"{rho_err:.3f}"),
                 "partial_rho_1mconf": ("" if pd.isna(rho_1mc) else f"{rho_1mc:.3f}"),
                 "n": len(g)})

out=pd.DataFrame(rows)
os.makedirs("tables", exist_ok=True)
out.to_csv("tables/race_partial_alignment_summary.csv", index=False)
print("[WROTE] tables/race_partial_alignment_summary.csv")
print(out.to_string(index=False) if len(out) else out)
