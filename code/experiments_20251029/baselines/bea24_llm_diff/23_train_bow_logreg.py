import pandas as pd, os
from sklearn.linear_model import LogisticRegression

ITEMS = "baselines/bea24_llm_diff/out/all_items_bea_filled.csv"
X_BOW = "baselines/bea24_llm_diff/out/x_bow.csv"
OUT   = "baselines/bea24_llm_diff/out/bow_vs_mvhmda.csv"

items = pd.read_csv(ITEMS)
X = pd.read_csv(X_BOW)

def enc(s: str):
    if not isinstance(s, str):
        return 1
    sl = s.lower()
    if "简单" in s or "easy" in sl: return 0
    if "困难" in s or "hard" in sl: return 2
    return 1

y = items["label"].apply(enc).values

n = len(items)
n_tr = int(n*0.8)
n_dv = int(n*0.9)

Xtr, ytr = X.iloc[:n_tr], y[:n_tr]
Xdv, ydv = X.iloc[n_tr:n_dv], y[n_tr:n_dv]
Xte, yte = X.iloc[n_dv:], y[n_dv:]

clf = LogisticRegression(max_iter=200, n_jobs=-1)
clf.fit(Xtr, ytr)

acc_tr = clf.score(Xtr, ytr)
acc_dv = clf.score(Xdv, ydv)
acc_te = clf.score(Xte, yte)

pd.DataFrame([
    {"dataset":"mixed","baseline":"BOW-logreg","split":"train","acc":acc_tr},
    {"dataset":"mixed","baseline":"BOW-logreg","split":"dev","acc":acc_dv},
    {"dataset":"mixed","baseline":"BOW-logreg","split":"test","acc":acc_te},
    {"dataset":"Eedi","baseline":"MV-HMDA (autonorm) gpt4o","split":"-","acc":0.70},
    {"dataset":"Eedi","baseline":"MV-HMDA (autonorm) mini","split":"-","acc":0.80},
    {"dataset":"RACE","baseline":"MV-HMDA proxy-5runs (qwen)","split":"-","acc":0.864},
]).to_csv(OUT, index=False)
print("✅ wrote", OUT)
print(f"acc: train={acc_tr:.4f} dev={acc_dv:.4f} test={acc_te:.4f}")
