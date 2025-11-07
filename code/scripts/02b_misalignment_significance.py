
#!/usr/bin/env python
# -*- coding: utf-8 -*-
import os, itertools
import pandas as pd, numpy as np
import matplotlib.pyplot as plt
from statsmodels.stats.proportion import proportion_confint, proportions_ztest
from statsmodels.stats.multitest import multipletests

def main():
    os.makedirs("tables", exist_ok=True)
    os.makedirs("figures", exist_ok=True)

    src = "tables/race_misalignment_summary.csv"
    if not os.path.exists(src):
        raise FileNotFoundError(f"{src} not found. Run scripts/02_misalignment_and_tau.py first.")

    df = pd.read_csv(src)
    # rates
    df["easy→model_hard_rate"] = df["#human_easy__model_hard"] / df["n_items"]
    df["hard→model_easy_rate"] = df["#human_hard__model_easy"] / df["n_items"]

    # 95% Wilson CI helper
    def add_ci(d, count_col, n_col, prefix):
        low, hi = proportion_confint(d[count_col], d[n_col], alpha=0.05, method="wilson")
        d[f"{prefix}_low95"] = low
        d[f"{prefix}_hi95"]  = hi
        return d

    df = add_ci(df, "#human_easy__model_hard", "n_items", "easy_ci")
    df = add_ci(df, "#human_hard__model_easy", "n_items", "hard_ci")

    # pretty csv
    pretty = df.copy()
    for c in ["easy→model_hard_rate","hard→model_easy_rate","easy_ci_low95","easy_ci_hi95","hard_ci_low95","hard_ci_hi95"]:
        pretty[c] = pretty[c].apply(lambda x: f"{x:.3f}")
    pretty = pretty.sort_values("easy→model_hard_rate", ascending=False)
    pretty.to_csv("tables/race_misalignment_with_ci.csv", index=False)
    print("[WROTE] tables/race_misalignment_with_ci.csv")

    # pairwise prop tests with Holm correction
    def pairwise_prop_tests(count_col, n_col, tag):
        rows=[]
        models = df["model"].tolist()
        for a,b in itertools.combinations(models, 2):
            da = df.loc[df["model"]==a].iloc[0]
            db = df.loc[df["model"]==b].iloc[0]
            count = np.array([da[count_col], db[count_col]], dtype=float)
            nobs  = np.array([da[n_col],   db[n_col]],   dtype=float)
            stat, p = proportions_ztest(count, nobs, alternative="two-sided")
            rows.append({"A":a,"B":b,"z":stat,"p_raw":p})
        out = pd.DataFrame(rows)
        if len(out):
            out["p_adj"] = multipletests(out["p_raw"], method="holm")[1]
            out["sig"]   = out["p_adj"].apply(lambda p: "***" if p<0.001 else ("**" if p<0.01 else ("*" if p<0.05 else "")))
        out = out.sort_values("p_adj", na_position="last")
        out.to_csv(f"tables/race_pairwise_{tag}_holm.csv", index=False)
        print(f"[WROTE] tables/race_pairwise_{tag}_holm.csv")
        return out

    pairwise_prop_tests("#human_easy__model_hard","n_items","easy_to_model_hard")
    pairwise_prop_tests("#human_hard__model_easy","n_items","hard_to_model_easy")

    # plot with CI
    order = df.sort_values("easy→model_hard_rate", ascending=False)["model"].tolist()
    D = df.set_index("model").loc[order]
    x = np.arange(len(order)); w = 0.38

    plt.figure(figsize=(7.2,4.6))
    # easy→model-hard
    y1 = D["easy→model_hard_rate"].values
    y1lo = y1 - D["easy_ci_low95"].values
    y1hi = D["easy_ci_hi95"].values - y1
    plt.bar(x - w/2, y1, width=w, label="Human-easy → Model-hard",
            yerr=[y1lo, y1hi], capsize=3)
    # hard→model-easy
    y2 = D["hard→model_easy_rate"].values
    y2lo = y2 - D["hard_ci_low95"].values
    y2hi = D["hard_ci_hi95"].values - y2
    plt.bar(x + w/2, y2, width=w, label="Human-hard → Model-easy",
            yerr=[y2lo, y2hi], capsize=3)

    plt.xticks(x, order)
    plt.ylabel("Rate")
    plt.title("RACE: Misalignment rates (95% Wilson CI)")
    plt.ylim(0, max((y1+y1hi).max(), (y2+y2hi).max())*1.15)
    plt.legend(frameon=False)
    plt.tight_layout()
    figp = "figures/race_misalignment_bars_wilson.png"
    plt.savefig(figp, dpi=160); plt.close()
    print("[WROTE]", figp)

    # LaTeX table
    latex = df.copy()
    latex["Easy→Hard (rate±)"] = latex.apply(
        lambda r: f"{r['easy→model_hard_rate']:.3f} [{r['easy_ci_low95']:.3f}, {r['easy_ci_hi95']:.3f}]",
        axis=1)
    latex["Hard→Easy (rate±)"] = latex.apply(
        lambda r: f"{r['hard→model_easy_rate']:.3f} [{r['hard_ci_low95']:.3f}, {r['hard_ci_hi95']:.3f}]",
        axis=1)
    latex = latex[["model","n_items","Easy→Hard (rate±)","Hard→Easy (rate±)"]].rename(
        columns={"model":"Model","n_items":"N"})
    tex = latex.to_latex(index=False, escape=False)
    open("tables/race_misalignment_wilson_table.tex","w").write(tex)
    print("[WROTE] tables/race_misalignment_wilson_table.tex")

    print("[DONE] 02b_misalignment_significance")

if __name__ == "__main__":
    main()
