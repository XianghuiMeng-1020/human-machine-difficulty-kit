import argparse, os, random
import pandas as pd
import numpy as np

RANDOM_SEED = 42
random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)

def to_matrix(df, use_diff=True):
    feats = []
    for _, r in df.iterrows():
        base = [
            1.0,
            r["prev_corr"],
            r["seq_len"],
            r["hist_acc"],
            r["seen_cur_item"],
        ]
        if use_diff:
            base += [
                r["hist_max_diff"],
                r["hist_avg_diff"],
            ]
        feats.append(base)
    X = np.array(feats, dtype=np.float32)
    y = df["label"].astype(np.float32).values
    return X, y

def train_logreg(X, y, lr=0.1, epochs=200):
    n, d = X.shape
    W = np.zeros(d, dtype=np.float32)
    for ep in range(epochs):
        logits = X @ W
        preds = 1.0 / (1.0 + np.exp(-logits))
        grad = (preds - y) @ X / n
        W -= lr * grad
    logits = X @ W
    preds = 1.0 / (1.0 + np.exp(-logits))
    acc = ((preds >= 0.5) == (y >= 0.5)).mean()
    return W, acc

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--feats", default="baselines/cl4kt_diff/ednet_u200_feats_strict.csv")
    ap.add_argument("--out_csv", default="baselines/cl4kt_diff/out/cl4kt_strict_vs_mvhmda.csv")
    args = ap.parse_args()

    df = pd.read_csv(args.feats)
    users = df["user_id"].unique().tolist()
    random.shuffle(users)
    n_train = int(len(users) * 0.7)
    train_users = set(users[:n_train])

    train_df = df[df["user_id"].isin(train_users)].reset_index(drop=True)
    test_df  = df[~df["user_id"].isin(train_users)].reset_index(drop=True)

    # ① 带 diff
    X_tr, y_tr = to_matrix(train_df, use_diff=True)
    W, _ = train_logreg(X_tr, y_tr, lr=0.2, epochs=200)
    X_te, y_te = to_matrix(test_df, use_diff=True)
    logits = X_te @ W
    preds = 1.0 / (1.0 + np.exp(-logits))
    acc_with = ((preds >= 0.5) == (y_te >= 0.5)).mean()

    # ② 不带 diff
    X_tr2, y_tr2 = to_matrix(train_df, use_diff=False)
    W2, _ = train_logreg(X_tr2, y_tr2, lr=0.2, epochs=200)
    X_te2, y_te2 = to_matrix(test_df, use_diff=False)
    logits2 = X_te2 @ W2
    preds2 = 1.0 / (1.0 + np.exp(-logits2))
    acc_nodiff = ((preds2 >= 0.5) == (y_te2 >= 0.5)).mean()

    # 写结果
    os.makedirs(os.path.dirname(args.out_csv), exist_ok=True)
    pd.DataFrame([
        {"dataset": "EdNet-KT1(u200)", "method": "CL4KT-strict + diff (no torch)", "acc": float(acc_with), "notes": "user-level split"},
        {"dataset": "EdNet-KT1(u200)", "method": "CL4KT-strict w/o diff (no torch)", "acc": float(acc_nodiff), "notes": "user-level split"},
        {"dataset": "EdNet-KT1(u200)", "method": "MV-HMDA-proxy (yours)", "acc": None, "notes": "ednet_labels_u200.csv"},
    ]).to_csv(args.out_csv, index=False, encoding="utf-8")

    print("✅ wrote", args.out_csv)
    print(f"test acc (with diff)   = {acc_with:.4f}")
    print(f"test acc (without diff)= {acc_nodiff:.4f}")
