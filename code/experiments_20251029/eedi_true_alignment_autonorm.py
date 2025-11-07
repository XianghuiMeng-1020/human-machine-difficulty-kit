import pandas as pd
import argparse, os, re

CAND_QID = ["qid", "question_id", "id", "_id", "\ufeffqid"]

def load_with_qid(path):
    df = pd.read_csv(path)
    qcol = None
    for c in CAND_QID:
        if c in df.columns:
            qcol = c
            break
    if qcol is None:
        raise ValueError(f"{path} 里没找到题目ID列, cols={list(df.columns)}")
    if qcol != "qid":
        df = df.rename(columns={qcol: "qid"})
    return df

def norm_h_to_m(s: str):
    if not isinstance(s, str):
        return None
    s = s.strip()
    s_low = s.lower()

    # 统一去掉 proxy / h / label 这些尾巴
    s_low = re.sub(r"(?:_?proxy|_?label|_?h)$", "", s_low)

    # 明确低曝光 / missing
    if any(k in s_low for k in ["低", "low", "missing", "none", "unk", "未标", "nohuman"]):
        return None

    # 中文三档
    if "简" in s_low:
        return "简单M"
    if "中" in s_low:
        return "中等M"
    if "难" in s_low:
        return "困难M"

    # 英文三档
    if "easy" in s_low:
        return "简单M"
    if "mid" in s_low or "medium" in s_low:
        return "中等M"
    if "hard" in s_low:
        return "困难M"

    # 实在认不出来就 None
    return None

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--proxy", required=True)
    ap.add_argument("--model_tags", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    proxy = load_with_qid(args.proxy)
    model = load_with_qid(args.model_tags)

    # 找到 proxy 里的标签列
    if "H_proxy" not in proxy.columns:
        for alt in ["human_label", "label", "H_label"]:
            if alt in proxy.columns:
                proxy = proxy.rename(columns={alt: "H_proxy"})
                break
    if "H_proxy" not in proxy.columns:
        raise ValueError(f"{args.proxy} 里没找到 H_proxy, cols={list(proxy.columns)}")

    proxy["M_from_H"] = proxy["H_proxy"].apply(norm_h_to_m)

    # 标准化 model 列
    if "M_tau" not in model.columns:
        for alt in ["model_label", "M_label", "pred_label", "model_tag"]:
            if alt in model.columns:
                model = model.rename(columns={alt: "M_tau"})
                break
    if "M_tau" not in model.columns:
        raise ValueError(f"{args.model_tags} 里没找到 M_tau, cols={list(model.columns)}")

    df = proxy.merge(model[["qid","M_tau"]], on="qid", how="inner")

    eval_mask = df["M_from_H"].notna()
    df_eval = df[eval_mask].copy()
    n_eval = len(df_eval)
    aligned = (df_eval["M_from_H"] == df_eval["M_tau"]).sum()
    ratio = aligned / n_eval if n_eval else 0.0

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    df.to_csv(args.out, index=False)
    print(f"✅ wrote {args.out}")
    print(f"rows_total={len(df)} rows_eval={n_eval} aligned={aligned} ratio={ratio:.4f}")
    if not n_eval:
        # 把我们到底看到了哪些原始标签也打出来
        print("⚠︎ 没有能归一化的标签，proxy 里的值是：", sorted(df["H_proxy"].dropna().unique().tolist())[:50])
