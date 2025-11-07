import os
import pandas as pd
from sklearn.linear_model import LogisticRegression

FEAT_CSV = "baselines/bea24_llm_diff/out/all_items_bea_feats.csv"
OUT_CSV  = "baselines/bea24_llm_diff/out/llm_feat_vs_mvhmda.csv"

df = pd.read_csv(FEAT_CSV)

def enc_label(s: str) -> int:
    if not isinstance(s, str):
        return 1
    sl = s.lower()
    if "简单" in s or "easy" in sl:
        return 0
    if "困难" in s or "hard" in sl:
        return 2
    return 1

df["y"] = df["label"].apply(enc_label)

X = df[["llm_diff_score", "llm_len", "llm_is_reasoning"]].values
y = df["y"].values

n = len(df)
n_train = int(n * 0.8)
n_dev   = int(n * 0.9)

Xtr, ytr = X[:n_train], y[:n_train]
Xdv, ydv = X[n_train:n_dev], y[n_train:n_dev]
Xte, yte = X[n_dev:], y[n_dev:]

clf = LogisticRegression(max_iter=1000, multi_class="multinomial")
clf.fit(Xtr, ytr)

acc_tr = clf.score(Xtr, ytr)
acc_dv = clf.score(Xdv, ydv)
acc_te = clf.score(Xte, yte)

rows = [
    {
        "dataset": "mixed (eedi+race+synthetic)",
        "baseline": "LLM-feat-only (BEA24-style)",
        "split": "train",
        "acc_or_align": float(acc_tr),
        "notes": "3-feat stub",
    },
    {
        "dataset": "mixed (eedi+race+synthetic)",
        "baseline": "LLM-feat-only (BEA24-style)",
        "split": "dev",
        "acc_or_align": float(acc_dv),
        "notes": "3-feat stub",
    },
    {
        "dataset": "mixed (eedi+race+synthetic)",
        "baseline": "LLM-feat-only (BEA24-style)",
        "split": "test",
        "acc_or_align": float(acc_te),
        "notes": "3-feat stub",
    },
    # 把你自己的 method 塞进来一张表里
    {
        "dataset": "Eedi",
        "baseline": "MV-HMDA (autonorm) gpt4o",
        "split": "-",
        "acc_or_align": 0.70,
        "notes": "analysis/eedi_proxy_x_gpt4o_true_autonorm.csv",
    },
    {
        "dataset": "Eedi",
        "baseline": "MV-HMDA (autonorm) gpt4o-mini",
        "split": "-",
        "acc_or_align": 0.80,
        "notes": "analysis/eedi_proxy_x_gpt4omini_true_autonorm.csv",
    },
    {
        "dataset": "RACE",
        "baseline": "MV-HMDA proxy-5runs (qwen)",
        "split": "-",
        "acc_or_align": 0.864,
        "notes": "paper_assets/mv-hmda_race/stage3_alignment_summary_race.csv",
    },
]

os.makedirs(os.path.dirname(OUT_CSV), exist_ok=True)
pd.DataFrame(rows).to_csv(OUT_CSV, index=False, encoding="utf-8")
print(f"✅ wrote {OUT_CSV}")
print(f"LLM-feat-only acc: train={acc_tr:.4f} dev={acc_dv:.4f} test={acc_te:.4f}")
