import pandas as pd, os

src = "analysis/global/global_alignment_table.csv"
df = pd.read_csv(src)

# 去掉之前那两行占位的 0.0
df = df[~((df["dataset"]=="Eedi") & (df["variant"]=="proxy-dense"))]

rows = [
    {
        "dataset": "Eedi",
        "variant": "proxy-dense-true",
        "model": "gpt4o",
        "alignment": 0.70,
        "notes": "autonorm, τ=0.8"
    },
    {
        "dataset": "Eedi",
        "variant": "proxy-dense-true",
        "model": "gpt4o-mini",
        "alignment": 0.80,
        "notes": "autonorm, τ=0.8"
    },
]

df2 = pd.concat([df, pd.DataFrame(rows)], ignore_index=True)

os.makedirs("analysis/global", exist_ok=True)
out = "analysis/global/global_alignment_table.csv"
df2.to_csv(out, index=False)
print("✅ updated", out)
