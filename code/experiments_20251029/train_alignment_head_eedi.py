import pandas as pd
import numpy as np
import os

# 读训练表
df = pd.read_csv("analysis/train_tabs/eedi_difficulty_triplet.csv")

# 只保留真的有人类 proxy 的
df = df[~df["H_as_M"].isna()].copy()

# 目标：把人侧三档变成 0/1/2
label_map = {"简单M": 0, "中等M": 1, "困难M": 2}
y = df["H_as_M"].map(label_map).values

# 特征设计（第一版很朴素）：
# 1) gpt4o 三档 one-hot
# 2) mini 三档 one-hot
# 3) 两个模型的置信度
def one_hot(col, cats=("简单M","中等M","困难M")):
    out = []
    for c in cats:
        out.append((col == c).astype(float).values.reshape(-1,1))
    return np.hstack(out)

X_parts = []

# gpt4o 的打标
X_parts.append(one_hot(df["M_gpt4o"].fillna("中等M")))
# mini 的打标
X_parts.append(one_hot(df["M_mini"].fillna("中等M")))
# 置信度
X_parts.append(df[["gpt4o_p_chosen","mini_p_chosen"]].fillna(0.0).values)

X = np.hstack(X_parts)  # shape (N, 3+3+2)=8

# 做一个最简单的多类 logistic (softmax)，手写一遍，避免装依赖
np.random.seed(0)
n_classes = 3
n_feats = X.shape[1]
W = np.zeros((n_feats, n_classes))  # 初始化
lr = 0.1
epochs = 300

def softmax(z):
    z = z - z.max(axis=1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=1, keepdims=True)

for ep in range(epochs):
    logits = X @ W  # (N,3)
    probs = softmax(logits)
    # one-hot label
    Y = np.zeros_like(probs)
    Y[np.arange(len(y)), y] = 1.0
    grad = X.T @ (probs - Y) / len(y)
    W -= lr * grad

# 训练完做一个简单评估
logits = X @ W
pred = logits.argmax(axis=1)
acc = (pred == y).mean()

# 每一档的 F1 很长，这里先给混淆
from collections import Counter
conf = np.zeros((3,3), dtype=int)
for yi, pi in zip(y, pred):
    conf[yi, pi] += 1

os.makedirs("analysis/alignment_head", exist_ok=True)
np.save("analysis/alignment_head/eedi_align_head_W.npy", W)

# 存报告
with open("analysis/alignment_head/eedi_align_head_report.txt","w",encoding="utf-8") as f:
    f.write(f"acc={acc:.4f}\n")
    f.write("confusion (rows=human/proxy, cols=pred):\n")
    for r in conf:
        f.write(",".join(map(str,r))+"\n")

print("✅ trained alignment head")
print(f"train acc={acc:.4f}")
print("confusion:\n", conf)
