import pandas as pd, os, random

SRC = "analysis/ednet_proxy_labels_full.csv"
OUT = "analysis/ednet_proxy_balanced_2k.csv"

df = pd.read_csv(SRC)

easy = df[df["H_proxy"]=="简单H_proxy"].copy()
mid  = df[df["H_proxy"]=="中等H_proxy"].copy()
hard = df[df["H_proxy"]=="困难H_proxy"].copy()
low  = df[df["H_proxy"]=="低曝光"].copy()

# 想要的目标大小（你可以再改）
target_easy = 1500
target_mid  = 400

if len(easy) > target_easy:
    easy = easy.sample(target_easy, random_state=0)
if len(mid) > target_mid:
    mid = mid.sample(target_mid, random_state=0)

balanced = pd.concat([easy, mid, hard, low], ignore_index=True)
balanced = balanced.sample(frac=1.0, random_state=0).reset_index(drop=True)

os.makedirs("analysis", exist_ok=True)
balanced.to_csv(OUT, index=False)
print("✅ wrote", OUT, "rows=", len(balanced))
print("   easy:", len(easy), "mid:", len(mid), "hard:", len(hard), "low:", len(low))
