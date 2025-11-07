import argparse
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

parser = argparse.ArgumentParser()
parser.add_argument("--cross", required=True, help="path to human_x_model csv")
parser.add_argument("--out", required=True, help="output figure path")
args = parser.parse_args()

# 读取交叉表
df = pd.read_csv(args.cross, index_col=0)

# 统一列顺序（防止缺列，且固定列序）
expected_cols = ["简单M", "中等M", "困难M"]
for col in expected_cols:
    if col not in df.columns:
        df[col] = 0
df = df[expected_cols]

plt.figure(figsize=(6, 4))
sns.heatmap(df, annot=True, fmt="d", cmap="YlOrBr")
plt.title("人类 vs 模型 难度交叉表")
plt.xlabel("模型难度 (M)")
plt.ylabel("人类难度 (H)")
plt.tight_layout()
plt.savefig(args.out, dpi=300)
print(f"✅ Saved: {args.out}")
