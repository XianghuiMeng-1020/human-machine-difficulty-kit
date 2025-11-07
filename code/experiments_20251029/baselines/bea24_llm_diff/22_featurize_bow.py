import os, re, pandas as pd
from collections import Counter

IN_CSV  = "baselines/bea24_llm_diff/out/all_items_bea_filled.csv"
OUT_X   = "baselines/bea24_llm_diff/out/x_bow.csv"
OUT_VOC = "baselines/bea24_llm_diff/out/vocab.txt"
TOPK_WANTED = 1000   # 想要的上限

df = pd.read_csv(IN_CSV)

def toks(s: str):
    return re.findall(r"[A-Za-z]+|\d+|[\u4e00-\u9fff]", s.lower())

# 1) 统计整个语料的词频
cnt = Counter()
for q in df["question"].fillna(""):
    cnt.update(toks(q))

# 实际能拿到的词数量
real_k = len(cnt)
topk = min(TOPK_WANTED, real_k) if real_k > 0 else 0
vocab = [w for w, _ in cnt.most_common(topk)]
idx = {w: i for i, w in enumerate(vocab)}

rows = []
for _, r in df.iterrows():
    q = r["question"] if isinstance(r["question"], str) else ""
    vec = [0] * topk
    for t in toks(q):
        if t in idx:
            vec[idx[t]] += 1
    # 确保这一行长度和列数一致
    rows.append(vec)

x_df = pd.DataFrame(rows, columns=[f"bow_{w}" for w in vocab])
os.makedirs(os.path.dirname(OUT_X), exist_ok=True)
x_df.to_csv(OUT_X, index=False)
with open(OUT_VOC, "w", encoding="utf-8") as f:
    for w in vocab:
        f.write(w + "\n")

print(f"✅ wrote {OUT_X} shape={x_df.shape}, vocab_size={topk}, from={IN_CSV}")
