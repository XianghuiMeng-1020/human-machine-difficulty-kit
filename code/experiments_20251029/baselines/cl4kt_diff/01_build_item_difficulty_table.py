import argparse, os
import pandas as pd

"""
我们已有：
  - analysis/ednet_labels_u200.csv   (item_id, H_proxy)
  - analysis/ednet_labels_u500.csv
  - analysis/ednet_labels_u1000.csv
这些是你刚才 scale 出来的 cov-aware proxy。

我们要生成一个标准的 item 表：
  item_id, diff_int, diff_str
让 KT 模型能直接 embed。
"""

LABEL_MAP = {
    "简单H_proxy": 0,
    "中等H_proxy": 1,
    "困难H_proxy": 2,
    # 你前面是「低曝光」「困难H_proxy」两档，这里我们把低曝光单独标成 -1
    "低曝光": -1,
}

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--labels", required=True, help="e.g. analysis/ednet_labels_u200.csv")
    ap.add_argument("--out", required=True, help="output csv")
    args = ap.parse_args()

    df = pd.read_csv(args.labels)
    # 统一列名
    if "item_id" not in df.columns:
        # 前面 flatten 出来的是 item_id
        raise SystemExit("❌ labels file must contain 'item_id'")
    if "H_proxy" not in df.columns:
        raise SystemExit("❌ labels file must contain 'H_proxy'")

    df["diff_int"] = df["H_proxy"].map(LABEL_MAP).fillna(-1).astype(int)
    df = df[["item_id", "H_proxy", "diff_int"]]

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    df.to_csv(args.out, index=False)
    print(f"✅ wrote {args.out} rows={len(df)}")
    print("dist:")
    print(df["diff_int"].value_counts(dropna=False))
