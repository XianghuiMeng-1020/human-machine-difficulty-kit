import pandas as pd
import argparse, os

if __name__=="__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--s1", default="analysis/ednet_proxy_labels.csv")
    ap.add_argument("--s2", default="analysis/ednet_proxy_labels_sample2.csv")
    ap.add_argument("--out", default="analysis/ednet_ed1_vs_ed2.csv")
    args = ap.parse_args()

    a = pd.read_csv(args.s1)
    b = pd.read_csv(args.s2)

    # 纯分布概要
    dist1 = a["H_proxy"].value_counts().rename("sample1_cnt")
    dist2 = b["H_proxy"].value_counts().rename("sample2_cnt")

    df = pd.concat([dist1, dist2], axis=1).fillna(0).astype(int)
    df["sample1_pct"] = (df["sample1_cnt"] / len(a)).round(4)
    df["sample2_pct"] = (df["sample2_cnt"] / len(b)).round(4)

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    df.to_csv(args.out)
    print("✅ wrote", args.out)
