import json, os
import pandas as pd

GLOBAL = "analysis/global/global_alignment_table.csv"
SEP_PATH = "analysis/synthetic_200/synthetic_separators.json"
Q_PATH = "synthetic/gen_questions_200.jsonl"

# 1) 读总题数
with open(Q_PATH, encoding="utf-8") as f:
    n_all = sum(1 for _ in f)

# 2) 读分歧题
with open(SEP_PATH, encoding="utf-8") as f:
    seps = json.load(f)
n_sep = len(seps)
rate = n_sep / n_all if n_all else 0.0

# 3) 写进 global
df = pd.read_csv(GLOBAL)

row = {
    "dataset": "Synthetic-200",
    "variant": "LLM-only divergence (3-model)",
    "model": "-",
    "alignment": rate,
    "notes": f"{n_sep}/{n_all} items had at least one model right/one wrong",
}

df = pd.concat([df, pd.DataFrame([row])], ignore_index=True)
df.to_csv(GLOBAL, index=False, encoding="utf-8")

print("✅ updated", GLOBAL, "with synthetic divergence =", f"{rate:.4f}")
