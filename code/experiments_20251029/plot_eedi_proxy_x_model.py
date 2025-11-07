import argparse, pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

if __name__=="__main__":
    ap=argparse.ArgumentParser()
    ap.add_argument("--csv", required=True)
    ap.add_argument("--out", required=True)
    args=ap.parse_args()

    df = pd.read_csv(args.csv)
    # 确保列齐
    for c in ["简单M","中等M","困难M"]:
        if c not in df.columns:
            df[c] = 0
    df = df[["H_proxy","简单M","中等M","困难M"]].set_index("H_proxy")

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    plt.figure(figsize=(5,3.5))
    sns.heatmap(df, annot=True, fmt="d", cmap="YlGnBu")
    plt.title("Proxy-H vs Model-M")
    plt.tight_layout()
    plt.savefig(args.out, dpi=300)
    print("✅ saved", args.out)
