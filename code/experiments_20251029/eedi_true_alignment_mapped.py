import pandas as pd
import argparse, os

CAND_QID = ["qid", "question_id", "id", "_id", "\ufeffqid"]

# H → M 的对齐字典
H2M = {
    "简单H": "简单M",
    "中等H": "中等M",
    "困难H": "困难M",
    # 有时候 proxy 可能写成英文/别名，这里兜一下
    "easy": "简单M",
    "medium": "中等M",
    "hard": "困难M",
}

def load_with_qid(path):
    df = pd.read_csv(path)
    qcol = None
    for c in CAND_QID:
        if c in df.columns:
            qcol = c
            break
    if qcol is None:
        raise ValueError(f"{path} 里没找到题目ID列，现有列: {list(df.columns)}")
    if qcol != "qid":
        df = df.rename(columns={qcol: "qid"})
    return df

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--proxy", required=True)
    ap.add_argument("--model_tags", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    proxy = load_with_qid(args.proxy)      # 有 qid, H_proxy
    model = load_with_qid(args.model_tags) # 有 qid, M_tau

    # 标准化 proxy 列名
    if "H_proxy" not in proxy.columns:
        # 有时叫 human_label / label
        for alt in ["human_label", "label", "H_label"]:
            if alt in proxy.columns:
                proxy = proxy.rename(columns={alt: "H_proxy"})
                break
    if "H_proxy" not in proxy.columns:
        raise ValueError(f"{args.proxy} 里没找到 H_proxy 列, cols={list(proxy.columns)}")

    # 把 H_proxy 映射成 M 风格
    proxy["M_from_H"] = proxy["H_proxy"].map(lambda x: H2M.get(str(x).strip(), None))

    # 标准化 model 列名
    if "M_tau" not in model.columns:
        for alt in ["model_label", "M_label", "pred_label"]:
            if alt in model.columns:
                model = model.rename(columns={alt: "M_tau"})
                break
    if "M_tau" not in model.columns:
        raise ValueError(f"{args.model_tags} 里没找到 M_tau 列, cols={list(model.columns)}")

    df = proxy.merge(model[["qid","M_tau"]], on="qid", how="inner")

    # 只在 H 能映射到 M 的样本上评估
    mask = df["M_from_H"].notna()
    df_eval = df[mask].copy()
    n_eval = len(df_eval)
    aligned = (df_eval["M_from_H"] == df_eval["M_tau"]).sum()
    ratio = aligned / n_eval if n_eval else 0.0

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    df.to_csv(args.out, index=False)
    print(f"✅ wrote {args.out} rows_total={len(df)} rows_eval={n_eval} aligned={aligned} ratio={ratio:.4f}")
