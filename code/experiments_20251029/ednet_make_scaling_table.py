import pandas as pd
import os

rows = []
cases = [
    ("u200",   "analysis/ednet_labels_u200.csv",   200),
    ("u200b",  "analysis/ednet_labels_u200b.csv",  200),
    ("u500",   "analysis/ednet_labels_u500.csv",   500),
    ("u1000",  "analysis/ednet_labels_u1000.csv",  1000),
]

for name, path, n_users in cases:
    if not os.path.exists(path):
        continue
    df = pd.read_csv(path)
    total = len(df)
    dist = df["H_proxy"].value_counts().to_dict()
    low  = dist.get("低曝光", 0)
    easy = dist.get("简单H_proxy", 0)
    mid  = dist.get("中等H_proxy", 0)
    hard = dist.get("困难H_proxy", 0)
    rows.append({
        "slice": name,
        "n_users": n_users,
        "n_items": total,
        "p_low":  low / total,
        "p_easy": easy / total,
        "p_mid":  mid / total,
        "p_hard": hard / total,
    })

out = pd.DataFrame(rows).sort_values("n_users")
os.makedirs("analysis/ednet_scale", exist_ok=True)
out.to_csv("analysis/ednet_scale/ednet_scaling.csv", index=False)
print("✅ wrote analysis/ednet_scale/ednet_scaling.csv")
print(out)
