import argparse, pandas as pd

if __name__=="__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True, help="aligned csv (already merged with H_proxy)")
    args = ap.parse_args()

    df = pd.read_csv(args.csv)
    # 保底
    if "H_proxy" not in df.columns:
        raise ValueError("need H_proxy in csv, run apply_alignment_head.py with --proxy")

    total = len(df)
    # 原始模型标签
    if "M_tau" in df.columns:
        base_align = (df["M_tau"] == df["H_proxy"].str.replace("H_proxy","M")).mean()
    else:
        base_align = float("nan")

    # head 后
    head_align = (df["aligned_h_like"] == df["H_proxy"].str.replace("H_proxy","M")).mean()

    print(f"Total={total}")
    print(f"base(M_tau)   alignment: {base_align:.4f}")
    print(f"head(aligned) alignment: {head_align:.4f}")
