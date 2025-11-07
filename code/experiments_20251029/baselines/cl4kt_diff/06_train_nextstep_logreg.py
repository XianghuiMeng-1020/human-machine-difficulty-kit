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
        probs = 1.0 / (1.0 + np.exp(-logits))
        grad = (probs - y) @ X / n
        W -= lr * grad
    return W

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--feats",   default="baselines/cl4kt_diff/ednet_u200_next_feats.csv")
    ap.add_argument("--out_csv", default="baselines/cl4kt_diff/out/cl4kt_nextstep_vs_mvhmda.csv")
    args = ap.parse_args()

    df = pd.read_csv(args.feats)
    users = df["user_id"].unique().tolist()
    random.shuffle(users)
    n_train = int(len(users) * 0.7)
    train_users = set(users[:n_train])

    train_df = df[df["user_id"].isin(train_users)].reset_index(drop=True)
    test_df  = df[~df["user_id"].isin(train_users)].reset_index(drop=True)

    # with diff
    Xtr, ytr = to_matrix(train_df, use_diff=True)
    W = train_logreg(Xtr, ytr, lr=0.2, epochs=200)
    Xte, yte = to_matrix(test_df, use_diff=True)
    p = 1.0 / (1.0 + np.exp(-(Xte @ W)))
    acc_with = ((p >= 0.5) == (yte >= 0.5)).mean()

    # no diff
    Xtr2, ytr2 = to_matrix(train_df, use_diff=False)
    W2 = train_logreg(Xtr2, ytr2, lr=0.2, epochs=200)
    Xte2, yte2 = to_matrix(test_df, use_diff=False)
    p2 = 1.0 / (1.0 + np.exp(-(Xte2 @ W2)))
    acc_nodiff = ((p2 >= 0.5) == (yte2 >= 0.5)).mean()

    os.makedirs(os.path.dirname(args.out_csv), exist_ok=True)
    pd.DataFrame([
        {"dataset": "EdNet-KT1(u200)", "method": "CL4KT-nextstep + diff", "acc": float(acc_with), "notes": "user-level split, predict t+1"},
        {"dataset": "EdNet-KT1(u200)", "method": "CL4KT-nextstep w/o diff", "acc": float(acc_nodiff), "notes": "user-level split, predict t+1"},
        {"dataset": "EdNet-KT1(u200)", "method": "MV-HMDA-proxy (yours)", "acc": None, "notes": "ednet_labels_u200.csv"},
    ]).to_csv(args.out_csv, index=False, encoding="utf-8")

    print("✅ wrote", args.out_csv)
    print(f"test acc (with diff)   = {acc_with:.4f}")
    print(f"test acc (without diff)= {acc_nodiff:.4f}")
