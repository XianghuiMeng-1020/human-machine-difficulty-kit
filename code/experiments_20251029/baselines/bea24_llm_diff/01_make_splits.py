import os
import pandas as pd
from sklearn.model_selection import train_test_split

RAW = "baselines/bea24_llm_diff/out/all_items_raw.csv"
OUTDIR = "baselines/bea24_llm_diff/out"
os.makedirs(OUTDIR, exist_ok=True)

df = pd.read_csv(RAW)

# 1) 清洗 label
# 我们现在的 label 里可能有：
# - "简单H_proxy" / "中等H_proxy" / "困难H_proxy"
# - "low" / "低曝光" 这种不能当 supervision 的
label_map = {
    "简单H_proxy": "easy",
    "中等H_proxy": "medium",
    "困难H_proxy": "hard",
    "简单H": "easy",
    "中等H": "medium",
    "困难H": "hard",
    "easy": "easy",
    "medium": "medium",
    "hard": "hard",
}
def normalize_label(x: str):
    if not isinstance(x, str):
        return None
    x = x.strip()
    if x in ("低曝光", "low", "low-coverage", "unknown", ""):
        return None
    return label_map.get(x, None)

df["label"] = df["H_proxy"].apply(normalize_label)
before = len(df)
df = df[df["label"].notna()].copy()
after = len(df)
print(f"[clean] {before} -> {after} rows kept for supervision")

# 2) question 字段可能为空，先填充一下，后面特征要用
df["question"] = df["question"].fillna("")

# 3) 我们按 source 分层拆分（eedi / race / synthetic）
frames = []
for src, g in df.groupby("source"):
    if len(g) < 10:
        # 太小的就全并进 train
        g["split"] = "train"
        frames.append(g)
        continue
    train, rest = train_test_split(g, test_size=0.2, random_state=42, shuffle=True)
    dev, test = train_test_split(rest, test_size=0.5, random_state=42, shuffle=True)
    train["split"] = "train"
    dev["split"] = "dev"
    test["split"] = "test"
    frames.extend([train, dev, test])

df2 = pd.concat(frames, ignore_index=True)
df2 = df2.sort_values(["source", "split", "qid"]).reset_index(drop=True)

out_path = os.path.join(OUTDIR, "all_items_splitted.csv")
df2.to_csv(out_path, index=False, encoding="utf-8")
print(f"✅ wrote {out_path}")

# 顺便各 split 打一份
for sp in ["train", "dev", "test"]:
    sp_df = df2[df2["split"] == sp].copy()
    sp_path = os.path.join(OUTDIR, f"{sp}.csv")
    sp_df.to_csv(sp_path, index=False, encoding="utf-8")
    print(f"✅ wrote {sp_path} rows={len(sp_df)}")
