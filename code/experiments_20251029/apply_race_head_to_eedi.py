import argparse, os
import pandas as pd
import numpy as np

def one_hot(col, cats=("简单M","中等M","困难M")):
    mats=[]
    for c in cats:
        mats.append((col==c).astype(float).values.reshape(-1,1))
    return np.hstack(mats)

if __name__=="__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--eedi_tab", default="analysis/train_tabs/eedi_difficulty_triplet.csv")
    ap.add_argument("--head", default="analysis/alignment_head_race/race_align_head_W.npy")
    ap.add_argument("--out", default="analysis/eedi_with_race_head.csv")
    args = ap.parse_args()

    df = pd.read_csv(args.eedi_tab)
    W = np.load(args.head)

    # eedi 这边只有 2 个模型：gpt4o, mini
    g4o_oh = one_hot(df["M_gpt4o"])
    mini_oh = one_hot(df["M_mini"])
    p_g4o = df["gpt4o_p_chosen"].values.reshape(-1,1)
    p_mini= df["mini_p_chosen"].values.reshape(-1,1)

    # race 头是 12 维：mini(3) + qwen(3) + deep(3) + p3
    # 我们要把 eedi 的 2 模型塞进去，然后剩下的全 0
    N = len(df)
    zero3 = np.zeros((N,3), dtype=float)

    # 我们映射成：mini ← eedi-mini, qwen ← eedi-gpt4o, deep ← 0
    X = np.hstack([
        mini_oh,          # to mini
        g4o_oh,           # to qwen slot
        zero3,            # deep slot
        p_mini,           # p_mini
        p_g4o,            # p_qwen
        np.zeros((N,1)),  # p_deep
    ])

    logits = X @ W
    logits = logits - logits.max(axis=1, keepdims=True)
    probs = np.exp(logits); probs /= probs.sum(axis=1, keepdims=True)
    pred = probs.argmax(axis=1)
    idx2label = {0:"简单M", 1:"中等M", 2:"困难M"}
    df["aligned_from_race"] = [idx2label[i] for i in pred]

    if "H_proxy" in df.columns:
        # 把 H_proxy 也转成 M 范式
        def h2m(h):
            if isinstance(h,str):
                if "简" in h: return "简单M"
                if "中" in h: return "中等M"
                if "难" in h: return "困难M"
            return None
        df["H_as_M"] = df["H_proxy"].apply(h2m)
        mask = df["H_as_M"].notna()
        base_g4o  = (df.loc[mask, "M_gpt4o"] == df.loc[mask, "H_as_M"]).mean()
        base_mini = (df.loc[mask, "M_mini"]  == df.loc[mask, "H_as_M"]).mean()
        head_aln  = (df.loc[mask, "aligned_from_race"] == df.loc[mask, "H_as_M"]).mean()
        print(f"[Eedi←RACE-head] N_eval={mask.sum()} gpt4o→{base_g4o:.4f} mini→{base_mini:.4f} head→{head_aln:.4f}")

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    df.to_csv(args.out, index=False, encoding="utf-8")
    print("✅ wrote", args.out, "rows=", len(df))
