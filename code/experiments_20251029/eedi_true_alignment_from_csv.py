import pandas as pd
import argparse, os

CAND_QID = ["qid", "question_id", "id", "_id", "\ufeffqid"]

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

    # 防止 model 里还有别的列，先只留需要的
    keep_model_cols = ["qid"]
    if "M_tau" in model.columns:
        keep_model_cols.append("M_tau")
    elif "model_label" in model.columns:
        model = model.rename(columns={"model_label": "M_tau"})
        keep_model_cols.append("M_tau")
    else:
        # 有的脚本写成了 "label"
        if "label" in model.columns:
            model = model.rename(columns={"label": "M_tau"})
            keep_model_cols.append("M_tau")
        else:
            raise ValueError(f"{args.model_tags} 里没找到模型标签列，现有列: {list(model.columns)}")
    model = model[keep_model_cols]

    df = proxy.merge(model, on="qid", how="inner")

    n = len(df)
    aligned = (df["H_proxy"] == df["M_tau"]).sum()
    ratio = aligned / n if n else 0.0

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    df.to_csv(args.out, index=False)
    print(f"✅ wrote {args.out} rows={n} aligned={aligned} ratio={ratio:.4f}")
