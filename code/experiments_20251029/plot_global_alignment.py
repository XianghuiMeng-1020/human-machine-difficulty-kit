import pandas as pd
import matplotlib.pyplot as plt
import os

df = pd.read_csv("analysis/global/global_alignment_table.csv")

plt.figure(figsize=(6, 3.8))

order = ["Eedi", "RACE", "EdNet-KT1"]
colors = {
    "Eedi": "tab:blue",
    "RACE": "tab:green",
    "EdNet-KT1": "tab:orange",
}

xs, ys, cs, labels = [], [], [], []

for _, row in df.iterrows():
    ds = row["dataset"]
    if ds not in order:
        continue
    x = order.index(ds)
    y = float(row["alignment"]) if pd.notna(row["alignment"]) else 0.0
    # label优先用model，没有就用variant
    if pd.notna(row.get("model")) and row["model"] != "-":
        label = f'{ds}-{row["model"]}'
    else:
        label = f'{ds}-{row.get("variant","")}'
    xs.append(x); ys.append(y); cs.append(colors.get(ds, "gray")); labels.append(label)

plt.scatter(xs, ys, c=cs)
for xi, yi, lb in zip(xs, ys, labels):
    plt.text(xi + 0.03, yi, lb, fontsize=7, va="center")

plt.xticks(range(len(order)), order)
plt.ylim(0, 1.05)
plt.ylabel("alignment")
plt.title("Human ↔ Model difficulty alignment (all datasets)")
plt.tight_layout()
os.makedirs("figs/global", exist_ok=True)
plt.savefig("figs/global/global_alignment.png", dpi=300)
print("✅ saved figs/global/global_alignment.png")
