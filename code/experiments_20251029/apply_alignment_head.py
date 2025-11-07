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
    ap.add_argument("--in_csv", required=True, help="like analysis/eedi_gpt4o_tau08_model_tags.csv")
    ap.add_argument("--out_csv", required=True, help="where to write aligned csv")
    ap.add_argument("--head", default="analysis/alignment_head/eedi_align_head_W.npy")
    # 如果这张表本身就有 proxy，我们也能一起输出，方便对齐
    ap.add_argument("--proxy", default="analysis/eedi_proxy_labels.csv")
    args = ap.parse_args()

    W = np.load(args.head)  # (8,3)
    df = pd.read_csv(args.in_csv)

    # 统一成 qid
    if "qid" not in df.columns:
        if "question_id" in df.columns:
            df = df.rename(columns={"question_id":"qid"})
        else:
            raise ValueError("can't find id column in input.")

    # 准备特征：
    # 这里我们的 head 是用 “gpt4o + mini + 两个置信度” 训练的，
    # 但现在我们可能只看到一张表（比如只有 gpt4o），怎么办？
    # ——我们就把“另一半”补成“中等M + 0.0”
    if "M_tau" in df.columns:
        m1 = df["M_tau"]
    else:
        # 有的表可能叫别的
        raise ValueError("input must have M_tau column")

    # 哪个模型？我们猜一下
    name = os.path.basename(args.in_csv)
    is_gpt4o = "gpt4o" in name
    is_mini  = "mini" in name

    # 构造 8 维特征
    # X = [gpt4o_onehot(3), mini_onehot(3), gpt4o_p, mini_p]
    # 情况1：这是 gpt4o 的表
    if is_gpt4o:
        g4o_oh = one_hot_col(m1)
        g4o_p  = df["p_chosen"].fillna(0.0).values.reshape(-1,1)
        mini_oh = one_hot_col(pd.Series(["中等M"]*len(df)))
        mini_p  = np.zeros((len(df),1))
    # 情况2：这是 mini 的表
    elif is_mini:
        mini_oh = one_hot_col(m1)
        mini_p  = df["p_chosen"].fillna(0.0).values.reshape(-1,1)
        g4o_oh  = one_hot_col(pd.Series(["中等M"]*len(df)))
        g4o_p   = np.zeros((len(df),1))
    else:
        # 都不是，就当成“只有一份模型标签”
        g4o_oh  = one_hot_col(m1)
        g4o_p   = df["p_chosen"].fillna(0.0).values.reshape(-1,1)
        mini_oh = one_hot_col(pd.Series(["中等M"]*len(df)))
        mini_p  = np.zeros((len(df),1))

    X = np.hstack([g4o_oh, mini_oh, g4o_p, mini_p])  # (N,8)
    # softmax
    logits = X @ W
    logits = logits - logits.max(axis=1, keepdims=True)
    probs = np.exp(logits); probs = probs / probs.sum(axis=1, keepdims=True)
    pred = probs.argmax(axis=1)

    idx2label = {0:"简单M", 1:"中等M", 2:"困难M"}
    df["aligned_h_like"] = [idx2label[i] for i in pred]

    # 如果给了 proxy，我们把 proxy 合进去，方便后面算acc
    if os.path.exists(args.proxy):
        proxy = pd.read_csv(args.proxy).rename(columns={"question_id":"qid"})
        df = df.merge(proxy[["qid","H_proxy"]], on="qid", how="left")

    os.makedirs(os.path.dirname(args.out_csv), exist_ok=True)
    df.to_csv(args.out_csv, index=False, encoding="utf-8")
    print("✅ wrote", args.out_csv, "rows=", len(df))
