
import pandas as pd, numpy as np, os

IN  = "tables/race_model_comparison_summary.csv"
OUT = "tables/race_model_comparison_summary.csv"
PRETTY = "tables/race_model_comparison_summary_pretty.csv"
LATEX = "tables/race_model_comparison_summary.tex"

def _coalesce_cols(df, base):
    x, y = base+"_x", base+"_y"
    if x in df.columns or y in df.columns:
        ax = df[x] if x in df.columns else None
        ay = df[y] if y in df.columns else None
        def coalesce(a,b):
            if a is None and b is None: return np.nan
            if a is None: return b
            if b is None: return a
            return b if (pd.notna(b)) else a
        df[base] = [coalesce(ax.iat[i] if ax is not None else None,
                             ay.iat[i] if ay is not None else None)
                    for i in range(len(df))]
        for c in (x,y):
            if c in df.columns: del df[c]
    return df

def main():
    if not os.path.exists(IN):
        print(f"[SKIP] {IN} not found"); return
    df = pd.read_csv(IN)

    for base in ["partial_rho_err","partial_rho_1mconf"]:
        df = _coalesce_cols(df, base)

    # numeric cast
    num_cols = ["Acc","MeanConf","ECE(15)","Brier","ROC_AUC","partial_rho_err","partial_rho_1mconf"]
    for c in num_cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    # reorder
    order = ["model","n","Acc","MeanConf","ECE(15)","Brier","ROC_AUC","partial_rho_err","partial_rho_1mconf"]
    order = [c for c in order if c in df.columns] + [c for c in df.columns if c not in order]
    df = df[order]

    # write numeric (machine-readable)
    df.to_csv(OUT, index=False)
    print("[WROTE]", OUT)

    # pretty version
    pretty = df.copy()
    def fmt3(x): 
        return "" if pd.isna(x) else f"{x:.3f}"
    for c in ["Acc","MeanConf","ECE(15)","Brier","ROC_AUC","partial_rho_err","partial_rho_1mconf"]:
        if c in pretty.columns:
            pretty[c] = pretty[c].map(fmt3)
    pretty.to_csv(PRETTY, index=False)
    print("[WROTE]", PRETTY)

    # LaTeX table
    # keep tidy column names for paper
    latex = pretty.rename(columns={
        "model":"Model","n":"N","Acc":"Acc","MeanConf":"MeanConf","ECE(15)":"ECE(15)","Brier":"Brier",
        "ROC_AUC":"ROC-AUC","partial_rho_err":"Partial ρ (err)","partial_rho_1mconf":"Partial ρ (1−conf)"
    })
    with open(LATEX,"w") as f:
        f.write(latex.to_latex(index=False, escape=True))
    print("[WROTE]", LATEX)

if __name__ == "__main__":
    main()
