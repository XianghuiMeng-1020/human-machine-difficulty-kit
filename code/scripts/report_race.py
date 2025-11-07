import os, argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from src.analysis.calibration import ece_bin, brier

def summarize_scores(df: pd.DataFrame) -> pd.DataFrame:
    g = df.groupby("question_id", as_index=False)
    out = g.apply(lambda x: pd.Series({
        "mean_p": x["p_correct"].mean(skipna=True),
        "std_p":  x["p_correct"].std(ddof=0, skipna=True)
    }))
    if "question_id" in out.columns:
        keep = ["question_id","mean_p","std_p"]
        out = out[[c for c in keep if c in out.columns]]
    return out

def datamap_counts(df: pd.DataFrame,
                   mean_easy=0.70, std_easy=0.15,
                   mean_hard=0.30, std_hard=0.15) -> dict:
    dm = summarize_scores(df)
    easy = ((dm["mean_p"] >= mean_easy) & (dm["std_p"] < std_easy)).sum()
    hard = ((dm["mean_p"] <  mean_hard) & (dm["std_p"] >= std_hard)).sum()
    impossible = ((dm["mean_p"] < mean_hard) & (dm["std_p"] < std_hard)).sum()
    ambiguous = len(dm) - easy - hard - impossible
    return {"Easy": int(easy), "Ambiguous": int(ambiguous),
            "Hard": int(hard), "Impossible": int(impossible)}

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scores", required=True)
    ap.add_argument("--outdir", required=True)
    ap.add_argument("--dm-thresholds", default="0.70,0.15,0.30,0.15")
    args = ap.parse_args()

    os.makedirs(args.outdir, exist_ok=True)
    df = pd.read_csv(args.scores)

    need = {"question_id","run_id","correct","p_chosen","p_correct","p_A","p_B","p_C","p_D"}
    missing = [c for c in need if c not in df.columns]
    if missing:
        raise RuntimeError(f"missing columns in scores: {missing}")

    y = df["correct"].astype(float).to_numpy()
    p = df["p_correct"].astype(float).to_numpy()
    overall = {
        "Overall Acc": float(np.nanmean(y)),
        "Mean Conf (chosen)": float(df["p_chosen"].mean(skipna=True)),
        "Mean Conf (correct)": float(df["p_correct"].mean(skipna=True)),
        "ECE(10)": float(ece_bin(y[~np.isnan(y)], p[~np.isnan(p)], 10)),
        "Brier": float(brier(y[~np.isnan(y)], p[~np.isnan(p)])),
        "Stable(identical across rounds)": int(
            (df.groupby("question_id")["chosen"].nunique(dropna=True) == 1).sum()
        ),
        "≤2-answers across rounds": int(
            (df.groupby("question_id")["chosen"].nunique(dropna=True) <= 2).sum()
        ),
        "n_rows": int(len(df)),
        "n_questions": int(df["question_id"].nunique()),
        "rounds": int(df["run_id"].nunique()),
    }

    me, se, mh, sh = map(float, args.dm_thresholds.split(","))
    dm_counts = datamap_counts(df, me, se, mh, sh)

    if "difficulty" in df.columns:
        by_human = (
            df.groupby("difficulty", as_index=False)
              .apply(lambda g: pd.Series({
                  "n": float(len(g)),
                  "Accuracy": float(g["correct"].mean(skipna=True)),
                  "Mean Conf (chosen)": float(g["p_chosen"].mean(skipna=True)),
                  "Mean Conf (correct)": float(g["p_correct"].mean(skipna=True)),
                  "Std Conf": float(g["p_correct"].std(ddof=0, skipna=True)),
              }))
        ).reset_index(drop=True)
    else:
        by_human = pd.DataFrame(columns=["difficulty","n","Accuracy",
                                         "Mean Conf (chosen)","Mean Conf (correct)","Std Conf"])

    pd.DataFrame([overall]).to_csv(os.path.join(args.outdir, "overall.csv"), index=False)
    by_human.to_csv(os.path.join(args.outdir, "by_human.csv"), index=False)
    pd.DataFrame([dm_counts]).to_csv(os.path.join(args.outdir, "datamap_counts.csv"), index=False)

    bins = np.linspace(0,1,11)
    df["bin"] = np.digitize(df["p_chosen"], bins, right=True)
    acc_b = df.groupby("bin")["correct"].mean()
    conf_b = df.groupby("bin")["p_chosen"].mean()
    plt.figure()
    plt.plot([0,1],[0,1],"--",linewidth=1)
    plt.plot(conf_b, acc_b, marker="o")
    plt.xlabel("Mean confidence"); plt.ylabel("Accuracy"); plt.title("Reliability")
    plt.savefig(os.path.join(args.outdir, "reliability.png"), bbox_inches="tight"); plt.close()

    summ = summarize_scores(df.dropna(subset=["p_correct","correct"]))
    plt.figure()
    plt.scatter(summ["mean_p"], summ["std_p"], s=8, alpha=0.6)
    plt.xlabel("Mean p_correct"); plt.ylabel("Std p_correct"); plt.title("Data Map (per question)")
    plt.savefig(os.path.join(args.outdir, "datamap.png"), bbox_inches="tight"); plt.close()

    with open(os.path.join(args.outdir,"summary.txt"),"w",encoding="utf-8") as f:
        f.write("== Overall ==\n")
        for k,v in overall.items():
            f.write(f"{k}: {v}\n")
        f.write("\n== Data Map counts ==\n")
        for k in ["Easy","Ambiguous","Hard","Impossible"]:
            f.write(f"{k}: {dm_counts.get(k,0)}\n")
        if len(by_human):
            f.write("\n== By human difficulty ==\n")
            f.write(by_human.to_string(index=False))

    print("Done.\n  Wrote:",
          os.path.join(args.outdir,"overall.csv"), "\n         ",
          os.path.join(args.outdir,"by_human.csv"), "\n         ",
          os.path.join(args.outdir,"datamap_counts.csv"), "\n         ",
          os.path.join(args.outdir,"summary.txt"), "\n         ",
          os.path.join(args.outdir,"reliability.png"), "\n         ",
          os.path.join(args.outdir,"datamap.png"))

if __name__ == "__main__":
    main()