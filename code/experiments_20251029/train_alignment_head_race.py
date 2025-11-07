import os, numpy as np, pandas as pd

# 1) 读我们刚刚做好的表
df = pd.read_csv("analysis/train_tabs/race_difficulty_triplet.csv")

# 2) 只保留标了 proxy 的
mask = df["H_as_M"].notna()
df = df[mask].reset_index(drop=True)

def one_hot(col, cats=("简单M","中等M","困难M")):
    mats=[]
    for c in cats:
        mats.append((col==c).astype(float).values.reshape(-1,1))
    return np.hstack(mats)

mini_oh = one_hot(df["M_mini"])
qwen_oh = one_hot(df["M_qwen"])
deep_oh = one_hot(df["M_deep"])

p_mini = df["p_mini"].values.reshape(-1,1)
p_qwen = df["p_qwen"].values.reshape(-1,1)
p_deep = df["p_deep"].values.reshape(-1,1)

X = np.hstack([mini_oh, qwen_oh, deep_oh, p_mini, p_qwen, p_deep])  # (N,12)

label_map = {"简单M":0, "中等M":1, "困难M":2}
y = df["H_as_M"].map(label_map).astype(int).values  # (N,)

# 3) 简单的多类 logistic，用近似的闭式：用一层线性 + softmax + 小步数梯度下降
N, D = X.shape
K = 3
W = np.zeros((D, K), dtype=float)

lr = 0.1
epochs = 80
for ep in range(epochs):
    logits = X @ W
    logits = logits - logits.max(axis=1, keepdims=True)
    exp = np.exp(logits)
    probs = exp / exp.sum(axis=1, keepdims=True)

    # one-hot y
    Y = np.zeros_like(probs)
    Y[np.arange(N), y] = 1.0

    grad = X.T @ (probs - Y) / N
    W -= lr * grad

# 4) 训练 acc
logits = X @ W
pred = logits.argmax(axis=1)
acc = (pred == y).mean()

# 混淆看一眼
conf = np.zeros((K,K), dtype=int)
for yi, pi in zip(y, pred):
    conf[yi, pi] += 1

os.makedirs("analysis/alignment_head_race", exist_ok=True)
np.save("analysis/alignment_head_race/race_align_head_W.npy", W)
with open("analysis/alignment_head_race/race_align_head_report.txt","w",encoding="utf-8") as f:
    f.write(f"acc={acc:.4f}\n")
    f.write("confusion (rows=H_proxy, cols=pred):\n")
    for r in conf:
        f.write(",".join(map(str,r))+"\n")

print("✅ trained race alignment head")
print(f"train acc={acc:.4f}")
print("confusion:\n", conf)
