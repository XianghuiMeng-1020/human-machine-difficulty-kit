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
    ap.add_argument("--g4o", required=True, help="analysis/eedi_gpt4o_tau08_model_tags.csv")
    ap.add_argument("--mini", required=True, help="analysis/eedi_gpt4omini_tau08_model_tags.csv")
    ap.add_argument("--proxy", required=True, help="analysis/eedi_proxy_labels.csv")
    ap.add_argument("--head", default="analysis/alignment_head/eedi_align_head_W.npy")
    ap.add_argument("--out", required=True, help="merged + aligned csv")
    args = ap.parse_args()

    W = np.load(args.head)

    g4o = pd.read_csv(args.g4o).rename(columns={"qid":"qid"})
    mini = pd.read_csv(args.mini).rename(columns={"qid":"qid"})
    proxy = pd.read_csv(args.proxy).rename(columns={"question_id":"qid"})

    # 先把三张表用 qid merge 到一行
    df = proxy.merge(g4o[["qid","M_tau","p_chosen"]].rename(
                        columns={"M_tau":"M_g4o","p_chosen":"p_g4o"}),
                     on="qid", how="left")
    df = df.merge(mini[["qid","M_tau","p_chosen"]].rename(
                        columns={"M_tau":"M_mini","p_chosen":"p_mini"}),
                  on="qid", how="left")

    # 缺的补成中等+0
    df["M_g4o"]  = df["M_g4o"].fillna("中等M")
    df["M_mini"] = df["M_mini"].fillna("中等M")
    df["p_g4o"]  = df["p_g4o"].fillna(0.0)
    df["p_mini"] = df["p_mini"].fillna(0.0)

    # 组装训练时的 8 维特征
    g4o_oh  = one_hot_col(df["M_g4o"])
    mini_oh = one_hot_col(df["M_mini"])
    g4o_p   = df["p_g4o"].values.reshape(-1,1)
    mini_p  = df["p_mini"].values.reshape(-1,1)
    X = np.hstack([g4o_oh, mini_oh, g4o_p, mini_p])

    # 前向
    logits = X @ W
    logits = logits - logits.max(axis=1, keepdims=True)
    probs = np.exp(logits); probs = probs / probs.sum(axis=1, keepdims=True)
    pred = probs.argmax(axis=1)
    idx2label = {0:"简单M", 1:"中等M", 2:"困难M"}
    df["aligned_h_like"] = [idx2label[i] for i in pred]

    # 生成 H_proxy 对应的 M 版本，方便比对
    def h_to_m(h):
        if isinstance(h, str):
            if "简单" in h: return "简单M"
            if "中等" in h: return "中等M"
            if "困难" in h: return "困难M"
        return None
    df["H_as_M"] = df["H_proxy"].apply(h_to_m)

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    df.to_csv(args.out, index=False, encoding="utf-8")
    print("✅ wrote", args.out, "rows=", len(df))

    # 顺手在这里就打印一下对齐情况
    mask = df["H_as_M"].notna()
    base_g4o  = (df.loc[mask, "M_g4o"]  == df.loc[mask, "H_as_M"]).mean()
    base_mini = (df.loc[mask, "M_mini"] == df.loc[mask, "H_as_M"]).mean()
    head_aln  = (df.loc[mask, "aligned_h_like"] == df.loc[mask, "H_as_M"]).mean()
    print(f"[report] N_eval={mask.sum()} gpt4o→{base_g4o:.4f} mini→{base_mini:.4f} head→{head_aln:.4f}")
