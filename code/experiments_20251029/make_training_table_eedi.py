import pandas as pd
import os

proxy = pd.read_csv("analysis/eedi_proxy_labels.csv")  # qid, H_proxy
g4o   = pd.read_csv("analysis/eedi_gpt4o_tau08_model_tags.csv")   # qid, M_tau
mini  = pd.read_csv("analysis/eedi_gpt4omini_tau08_model_tags.csv")  # qid, M_tau

df = proxy.merge(g4o.rename(columns={"M_tau": "M_gpt4o"}), on="qid", how="left")
df = df.merge(mini.rename(columns={"M_tau": "M_mini"}), on="qid", how="left")

def h_to_m(h):
    if not isinstance(h, str):
        return None
    h = h.strip()
    if "易" in h or "简" in h or "easy" in h.lower():
        return "简单M"
    if "中" in h or "mid" in h.lower() or "medium" in h.lower():
        return "中等M"
    if "难" in h or "hard" in h.lower():
        return "困难M"
    return None

df["H_as_M"] = df["H_proxy"].apply(h_to_m)

os.makedirs("analysis/train_tabs", exist_ok=True)
out = "analysis/train_tabs/eedi_difficulty_triplet.csv"
df.to_csv(out, index=False)
print("✅ wrote", out, "rows=", len(df))
print(df.head(10).to_string(index=False))
