import os, numpy as np, pandas as pd

df = pd.read_csv("analysis/joint/eedi_race_joint.csv")

# 只要有 H_as_M 的
mask = df["H_as_M"].notna()
df = df[mask].reset_index(drop=True)

def one_hot_label(col, cats=("简单M","中等M","困难M")):
    mats=[]
    for c in cats:
        mats.append((col==c).astype(float).values.reshape(-1,1))
    return np.hstack(mats)

# 3 个 slot
slot1_oh = one_hot_label(df["slot1_label"])
slot2_oh = one_hot_label(df["slot2_label"])
slot3_oh = one_hot_label(df["slot3_label"])

slot1_p = df["slot1_p"].fillna(0.0).values.reshape(-1,1)
slot2_p = df["slot2_p"].fillna(0.0).values.reshape(-1,1)
slot3_p = df["slot3_p"].fillna(0.0).values.reshape(-1,1)

# dataset one-hot
is_eedi = (df["dataset"]=="eedi").astype(float).values.reshape(-1,1)
is_race = (df["dataset"]=="race").astype(float).values.reshape(-1,1)

X = np.hstack([
    slot1_oh, slot1_p,
    slot2_oh, slot2_p,
    slot3_oh, slot3_p,
    is_eedi, is_race
])  # (N, 14)

label_map = {"简单M":0, "中等M":1, "困难M":2}
y = df["H_as_M"].map(label_map).astype(int).values

N, D = X.shape
K = 3
W = np.zeros((D, K), dtype=float)

lr = 0.1
epochs = 100
for ep in range(epochs):
    logits = X @ W
    logits = logits - logits.max(axis=1, keepdims=True)
    exp = np.exp(logits)
    probs = exp / exp.sum(axis=1, keepdims=True)

    Y = np.zeros_like(probs)
    Y[np.arange(N), y] = 1.0

    grad = X.T @ (probs - Y) / N
    W -= lr * grad

# 评估：分开看 eedi / race
logits = X @ W
pred = logits.argmax(axis=1)
acc_all = (pred == y).mean()

df["pred_idx"] = pred
idx2label = {0:"简单M", 1:"中等M", 2:"困难M"}
df["pred_label"] = [idx2label[i] for i in pred]

acc_eedi = (df[df["dataset"]=="eedi"]["pred_idx"] == df[df["dataset"]=="eedi"]["H_as_M"].map(label_map)).mean()
acc_race = (df[df["dataset"]=="race"]["pred_idx"] == df[df["dataset"]=="race"]["H_as_M"].map(label_map)).mean()

os.makedirs("analysis/alignment_head_joint", exist_ok=True)
np.save("analysis/alignment_head_joint/joint_align_head_W.npy", W)
df.to_csv("analysis/alignment_head_joint/joint_train_pred.csv", index=False, encoding="utf-8")

print("✅ trained joint head")
print(f"all={acc_all:.4f}  eedi={acc_eedi:.4f}  race={acc_race:.4f}")
