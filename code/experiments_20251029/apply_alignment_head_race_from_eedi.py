import argparse, os
import pandas as pd
import numpy as np

def one_hot_col(series, cats=("简单M","中等M","困难M")):
    mats = []
    for c in cats:
        mats.append((series == c).astype(float).values.reshape(-1,1))
    return np.hstack(mats)

if __name__=="__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--race_tab", default="analysis/train_tabs/race_difficulty_triplet.csv")
    ap.add_argument("--head", default="analysis/alignment_head/eedi_align_head_W.npy")
    ap.add_argument("--out", default="analysis/race_with_eedi_head.csv")
    args = ap.parse_args()

    W = np.load(args.head)
    df = pd.read_csv(args.race_tab)

    # 只取两个模型：mini + qwen
    # （和 eedi 的结构一致：3 one-hot + 3 one-hot + p + p = 8 维）
    mini_oh = one_hot_col(df["M_mini"])
    qwen_oh = one_hot_col(df["M_qwen"])
    p_mini = df["p_mini"].values.reshape(-1,1)
    p_qwen = df["p_qwen"].values.reshape(-1,1)

    X = np.hstack([mini_oh, qwen_oh, p_mini, p_qwen])

    logits = X @ W
    logits = logits - logits.max(axis=1, keepdims=True)
    probs = np.exp(logits); probs /= probs.sum(axis=1, keepdims=True)
    pred = probs.argmax(axis=1)
    idx2label = {0:"简单M", 1:"中等M", 2:"困难M"}
    df["aligned_h_like"] = [idx2label[i] for i in pred]

    # 有 H_as_M 的就算对齐
    mask = df["H_as_M"].notna()
    base_mini = (df.loc[mask, "M_mini"] == df.loc[mask, "H_as_M"]).mean()
    base_qwen = (df.loc[mask, "M_qwen"] == df.loc[mask, "H_as_M"]).mean()
    head_aln  = (df.loc[mask, "aligned_h_like"] == df.loc[mask, "H_as_M"]).mean()

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    df.to_csv(args.out, index=False, encoding="utf-8")
    print("✅ wrote", args.out, "rows=", len(df))
    print(f"[RACE←Eedi-head] N_eval={mask.sum()} mini→{base_mini:.4f} qwen→{base_qwen:.4f} head→{head_aln:.4f}")
