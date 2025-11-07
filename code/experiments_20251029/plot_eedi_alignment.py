import argparse, os, csv
import pandas as pd
import matplotlib.pyplot as plt

if __name__=="__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default="analysis/eedi_all_alignment.csv")
    ap.add_argument("--out_dir", default="figs/eedi_alignment")
    ap.add_argument("--topk", type=int, default=20)
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    df = pd.read_csv(args.csv)

    # 有的行可能空字符串，先处理
    def to_float(x):
        try:
            return float(x)
        except:
            return None

    df["align_tau080_f"] = df["align_tau080"].apply(to_float)
    df["align_tau090_f"] = df["align_tau090"].apply(to_float)

    # 1) 画 τ=0.80 的分布
    plt.figure(figsize=(6,4))
    vals = [v for v in df["align_tau080_f"].tolist() if v is not None]
    plt.hist(vals, bins=10)
    plt.title("Eedi human–model alignment (τ=0.80)")
    plt.xlabel("alignment ratio")
    plt.ylabel("count")
    plt.tight_layout()
    out_hist = os.path.join(args.out_dir, "eedi_align_tau080_hist.png")
    plt.savefig(out_hist, dpi=300)
    print("✅ saved", out_hist)

    # 2) 导出最不对齐的前 topk
    df_bad = df.dropna(subset=["align_tau080_f"]).sort_values("align_tau080_f", ascending=True).head(args.topk)
    bad_csv = os.path.join(args.out_dir, f"eedi_align_tau080_bottom{args.topk}.csv")
    df_bad.to_csv(bad_csv, index=False)
    print("✅ wrote", bad_csv)

    # 3) 简单打印一个 summary，给后续写 paper 用
    if len(vals) > 0:
        print("—— summary (τ=0.80) ——")
        print("N dirs:", len(vals))
        print("min:", min(vals))
        print("max:", max(vals))
        print("mean:", sum(vals)/len(vals))
