import pandas as pd, numpy as np, os

SRC = "baselines/bea24_llm_diff/out/all_items_bea_feats.csv"
df = pd.read_csv(SRC)

# 把 label 统一到 3 类
def norm_lab(x):
    x = str(x)
    if "简" in x or "easy" in x.lower():
        return 0
    if "中" in x or "mid" in x.lower():
        return 1
    if "难" in x or "hard" in x.lower():
        return 2
    return 1

df["y"] = df["target_diff"].apply(norm_lab)

X = df[["n_tok","n_char","n_sent","n_opts","opt_len_var"]].values.astype(np.float32)
X = np.concatenate([np.ones((len(X),1),dtype=np.float32), X], axis=1)
y = df["y"].values

# very small train/test split
idx = np.arange(len(df))
np.random.shuffle(idx)
n_tr = int(len(idx)*0.8)
tr_idx, te_idx = idx[:n_tr], idx[n_tr:]

Xtr, ytr = X[tr_idx], y[tr_idx]
Xte, yte = X[te_idx], y[te_idx]

W = np.zeros((X.shape[1], 3), dtype=np.float32)
lr = 0.1
for ep in range(200):
    logits = Xtr @ W
    logits -= logits.max(axis=1, keepdims=True)
    exp = np.exp(logits)
    probs = exp / exp.sum(axis=1, keepdims=True)
    onehot = np.eye(3)[ytr]
    grad = Xtr.T @ (probs - onehot) / len(Xtr)
    W -= lr * grad

pred = (Xte @ W).argmax(axis=1)
acc = (pred == yte).mean()
os.makedirs("baselines/bea24_llm_diff/out", exist_ok=True)
np.save("baselines/bea24_llm_diff/out/bea_like_logreg_W.npy", W)
print(f"✅ BEA-like logreg acc={acc:.4f} on {len(yte)} items")
