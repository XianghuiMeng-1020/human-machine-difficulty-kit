import os
import pandas as pd
import numpy as np

BASE = "baselines/bea24_llm_diff/out"
train = pd.read_csv(os.path.join(BASE, "train_feats.csv"))
dev   = pd.read_csv(os.path.join(BASE, "dev_feats.csv"))
test  = pd.read_csv(os.path.join(BASE, "test_feats.csv"))

# 我们在 02 里已经把目标编码成 y ∈ {0,1,2} 了
FEATS = ["q_len", "q_words", "q_has_num", "q_avg_tok"]

def to_xy(df):
    X = df[FEATS].values.astype(np.float32)
    y = df["y"].values.astype(int)
    return X, y

Xtr, ytr = to_xy(train)
Xdv, ydv = to_xy(dev)
Xte, yte = to_xy(test)

n_feat = Xtr.shape[1]
n_cls  = 3

# 线性头：logits = XW + b
W = np.zeros((n_feat, n_cls), dtype=np.float32)
b = np.zeros((n_cls,), dtype=np.float32)

def softmax(z):
    z = z - z.max(axis=1, keepdims=True)
    ez = np.exp(z)
    return ez / ez.sum(axis=1, keepdims=True)

lr = 0.05
epochs = 200

for ep in range(1, epochs+1):
    logits = Xtr @ W + b
    probs  = softmax(logits)
    # one-hot
    onehot = np.eye(n_cls)[ytr]
    grad_logits = (probs - onehot) / len(Xtr)

    grad_W = Xtr.T @ grad_logits
    grad_b = grad_logits.sum(axis=0)

    W -= lr * grad_W
    b -= lr * grad_b

    if ep % 50 == 0:
        pred = probs.argmax(axis=1)
        acc  = (pred == ytr).mean()
        print(f"ep {ep:3d} train-acc={acc:.4f}")

# 评估函数
def eval_split(name, X, y, src):
    logits = X @ W + b
    probs  = softmax(logits)
    pred   = probs.argmax(axis=1)
    overall = (pred == y).mean()
    print(f"[{name}] overall acc={overall:.4f}  N={len(y)}")
    # 按 source 再拆
    df = pd.DataFrame({
        "pred": pred,
        "gold": y,
        "source": src
    })
    for s in df["source"].unique():
        sub = df[df["source"]==s]
        acc_s = (sub["pred"] == sub["gold"]).mean()
        print(f"   - {s:8s} acc={acc_s:.4f}  N={len(sub)}")
    return probs, pred

# train/dev/test 分别评估
probs_tr, pred_tr = eval_split("train", Xtr, ytr, train["source"])
probs_dv, pred_dv = eval_split("dev",   Xdv, ydv, dev["source"])
probs_te, pred_te = eval_split("test",  Xte, yte, test["source"])

# 把 test 的预测存下来，后面 04 要用
test_out = test.copy()
test_out["pred_id"]    = pred_te
MAP_BACK = {0:"简单H_proxy", 1:"中等H_proxy", 2:"困难H_proxy"}
test_out["pred_label"] = test_out["pred_id"].map(MAP_BACK)
test_out["p_easy"]     = probs_te[:,0]
test_out["p_medium"]   = probs_te[:,1]
test_out["p_hard"]     = probs_te[:,2]

os.makedirs(BASE, exist_ok=True)
out_csv = os.path.join(BASE, "bea24_stub_preds.csv")
test_out.to_csv(out_csv, index=False, encoding="utf-8")
print(f"✅ wrote {out_csv}")
