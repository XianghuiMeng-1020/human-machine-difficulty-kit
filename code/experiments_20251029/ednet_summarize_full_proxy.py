import pandas as pd
import os

IN = "analysis/ednet_proxy_labels_full.csv"
OUT = "analysis/ednet_full_summary.csv"

df = pd.read_csv(IN)

n_items = len(df)
dist = df["H_proxy"].value_counts().to_dict()

# 看一下 acc / err 的分位数，方便论文里说 “95% 的题 err < …”
q = df["err"].quantile([0.5, 0.75, 0.9, 0.95, 0.99]).to_dict()

rows = []
rows.append({"key": "n_items", "value": n_items})
for k, v in dist.items():
    rows.append({"key": f"dist_{k}", "value": v})
for k, v in q.items():
    rows.append({"key": f"err_q{int(k*100)}", "value": float(v)})

os.makedirs("analysis", exist_ok=True)
pd.DataFrame(rows).to_csv(OUT, index=False)
print("✅ wrote", OUT)
print("items:", n_items)
print("dist:", dist)
print("err quantiles:", q)
