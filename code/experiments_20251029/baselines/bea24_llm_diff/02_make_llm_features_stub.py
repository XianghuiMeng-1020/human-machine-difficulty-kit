import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

BASE = "baselines/bea24_llm_diff/out"
SRC  = os.path.join(BASE, "all_items_raw_filled.csv")
OUTDIR = BASE

df = pd.read_csv(SRC)

# 1) 简单 split（保持源域分布）
train_df, tmp = train_test_split(df, test_size=0.2, random_state=42, stratify=df["source"])
dev_df, test_df = train_test_split(tmp, test_size=0.5, random_state=42, stratify=tmp["source"])

train_df["split"] = "train"
dev_df["split"]   = "dev"
test_df["split"]  = "test"

df = pd.concat([train_df, dev_df, test_df], ignore_index=True)

def feat_len(s: str) -> int:
    return len(str(s))

def feat_words(s: str) -> int:
    return len(str(s).split())

def feat_has_num(s: str) -> int:
    return 1 if any(ch.isdigit() for ch in str(s)) else 0

df["q_len"] = df["question"].apply(feat_len)
df["q_words"] = df["question"].apply(feat_words)
df["q_has_num"] = df["question"].apply(feat_has_num)
df["q_avg_tok"] = (df["q_len"] / (df["q_words"] + 1e-5)).round(3)

# label -> y
MAP = {
    "简单H_proxy": 0,
    "中等H_proxy": 1,
    "困难H_proxy": 2,
    # RACE 那边 middle/high 我们先粗暴对一对
    "middleH_proxy": 1,
    "highH_proxy": 2,
}
df["y"] = df["H_proxy"].map(MAP).fillna(1).astype(int)

os.makedirs(OUTDIR, exist_ok=True)
df.to_csv(os.path.join(OUTDIR, "all_items_with_stub_feats.csv"), index=False, encoding="utf-8")
df[df["split"]=="train"].to_csv(os.path.join(OUTDIR, "train_feats.csv"), index=False, encoding="utf-8")
df[df["split"]=="dev"].to_csv(os.path.join(OUTDIR, "dev_feats.csv"), index=False, encoding="utf-8")
df[df["split"]=="test"].to_csv(os.path.join(OUTDIR, "test_feats.csv"), index=False, encoding="utf-8")

print(f"✅ wrote {OUTDIR}/all_items_with_stub_feats.csv rows={len(df)}")
print(f"✅ wrote {OUTDIR}/train_feats.csv rows={(df['split']=='train').sum()}")
print(f"✅ wrote {OUTDIR}/dev_feats.csv rows={(df['split']=='dev').sum()}")
print(f"✅ wrote {OUTDIR}/test_feats.csv rows={(df['split']=='test').sum()}")
