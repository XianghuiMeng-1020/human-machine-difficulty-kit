import os
import pandas as pd
import matplotlib.pyplot as plt

csv_path = "paper_assets/RESULTS_GLOBAL_BASELINES.csv"
df = pd.read_csv(csv_path)

# 为了美观，名字稍微变一下
rename = {
    "MV-HMDA (autonorm) gpt4o": "Ours (Eedi, gpt4o)",
    "MV-HMDA (autonorm) mini": "Ours (Eedi, mini)",
    "proxy-5runs": "Ours (RACE)",
    "BEA24-LLM-feat (logreg)": "LLM-feat",
    "BEA24-BOW (templated)": "BOW",
}

df["nice_name"] = df.apply(
    lambda r: rename.get(str(r["variant"]), str(r["variant"])), axis=1
)

plt.figure(figsize=(8, 4))

# 每个 dataset 独立画一下
datasets = df["dataset"].unique().tolist()
xpos = range(len(df))

# 简单画：其实就是一个条形图
plt.bar(range(len(df)), df["alignment"])

plt.xticks(range(len(df)), df["nice_name"], rotation=45, ha="right")
plt.ylim(0, 1.05)
plt.ylabel("alignment / accuracy")
plt.title("Human–Model difficulty alignment vs baselines")

os.makedirs("figs/paper", exist_ok=True)
plt.tight_layout()
plt.savefig("figs/paper/global_baselines.png", dpi=300)
print("✅ saved figs/paper/global_baselines.png")
