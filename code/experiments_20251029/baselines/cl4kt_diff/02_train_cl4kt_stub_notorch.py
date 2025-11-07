import argparse, os
import pandas as pd
import numpy as np

def load_sessions(sessions_csv, items_csv, max_len=100):
    """把 session + item 难度 拼成一条条训练样本"""
    sess = pd.read_csv(sessions_csv)
    items = pd.read_csv(items_csv)
    # items 里有: item_id, diff_int
    diff_map = dict(zip(items["item_id"].astype(str), items["diff_int"].astype(int)))

    X_feats = []
    y = []
    for row in sess.itertuples(index=False):
        its = str(row.items).split()
        cors = str(row.corrects).split()
        if not its:
            continue

        # 最后一题→目标
        last_it = its[-1]
        last_corr = int(float(cors[-1]))

        # 上一题是否做对 → 简单 KT 信号
        if len(its) >= 2:
            prev_corr = int(float(cors[-2]))
        else:
            prev_corr = 0

        # item 难度
        d = diff_map.get(last_it, -1)
        if d < 0:
            d = 3  # 用第4个桶表示“没难度”
        diff_onehot = [0, 0, 0, 0]
        diff_onehot[d] = 1

        # 序列长度特征
        seq_len = min(len(its), max_len) / max_len

        # item id 做个 hash → 稍微有点区分度
        it_hash = hash(last_it) % 997
        it_norm = it_hash / 997.0

        # 特征结构：
        # [1 , diff0, diff1, diff2, diff3, prev_corr, seq_len, it_norm]
        feats = [1.0] + diff_onehot + [prev_corr, seq_len, it_norm]
        X_feats.append(feats)
        y.append(last_corr)

    X = np.array(X_feats, dtype=np.float32)
    y = np.array(y, dtype=np.float32)
    return X, y

def train_logistic(X, y, lr=0.1, epochs=200):
    n, d = X.shape
    W = np.zeros(d, dtype=np.float32)
    for ep in range(epochs):
        logits = X @ W
        preds = 1.0 / (1.0 + np.exp(-logits))
        grad = (preds - y) @ X / n
        W -= lr * grad

        if (ep + 1) % 50 == 0:
            acc = ((preds >= 0.5) == (y >= 0.5)).mean()
            loss = -(y * np.log(preds + 1e-8) + (1 - y) * np.log(1 - preds + 1e-8)).mean()
            print(f"[ep {ep+1}] loss={loss:.4f} acc={acc:.4f}")

    # final
    logits = X @ W
    preds = 1.0 / (1.0 + np.exp(-logits))
    acc = ((preds >= 0.5) == (y >= 0.5)).mean()
    return W, acc

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--sessions", default="baselines/cl4kt_diff/ednet_u200_sessions.csv")
    ap.add_argument("--items",    default="baselines/cl4kt_diff/ednet_u200_items.csv")
    ap.add_argument("--out_dir",  default="baselines/cl4kt_diff/out")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    # ① 带 difficulty
    X, y = load_sessions(args.sessions, args.items)
    print(f"[with diff] loaded X={X.shape}, y={y.shape}")
    W_with, acc_with = train_logistic(X, y, lr=0.2, epochs=200)

    # ② 不带 difficulty：砍掉 diff 的 4 维
    # 结构是: 0:[bias], 1-4:diff, 5:prev_corr, 6:seq_len, 7:it_norm
    X_nodiff = np.concatenate([X[:, :1], X[:, 5:]], axis=1)
    print(f"[no  diff] loaded X={X_nodiff.shape}, y={y.shape}")
    _, acc_nodiff = train_logistic(X_nodiff, y, lr=0.2, epochs=200)

    out_csv = os.path.join(args.out_dir, "cl4kt_stub_vs_mvhmda.csv")
    pd.DataFrame([
        {"dataset": "EdNet-KT1(u200)", "method": "CL4KT-stub + diff (no torch)", "acc": float(acc_with)},
        {"dataset": "EdNet-KT1(u200)", "method": "CL4KT-stub w/o diff (no torch)", "acc": float(acc_nodiff)},
        {"dataset": "EdNet-KT1(u200)", "method": "MV-HMDA-proxy (yours)", "acc": None},
    ]).to_csv(out_csv, index=False, encoding="utf-8")
    print("✅ wrote", out_csv)
    print(f"→ with diff acc = {acc_with:.4f}")
    print(f"→ no  diff acc = {acc_nodiff:.4f}")
