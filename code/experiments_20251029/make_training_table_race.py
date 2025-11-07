import os, pandas as pd

proxy = pd.read_csv("paper_assets/mv-hmda_race/stage2_proxy_labels_race.csv")
mini  = pd.read_csv("paper_assets/mv-hmda_race/stage2_model_tags_race_gpt4omini_tau08.csv")
qwen  = pd.read_csv("paper_assets/mv-hmda_race/stage2_model_tags_race_qwen3next80b_tau08.csv")
deep  = pd.read_csv("paper_assets/mv-hmda_race/stage2_model_tags_race_deepseekv3_tau08.csv")

# 统一列名
mini = mini.rename(columns={"qid":"qid","M_tau":"M_mini","p_chosen":"p_mini"})
qwen = qwen.rename(columns={"qid":"qid","M_tau":"M_qwen","p_chosen":"p_qwen"})
deep = deep.rename(columns={"qid":"qid","M_tau":"M_deep","p_chosen":"p_deep"})
proxy = proxy.rename(columns={"question_id":"qid"})

df = proxy.merge(mini[["qid","M_mini","p_mini"]], on="qid", how="left")
df = df.merge(qwen[["qid","M_qwen","p_qwen"]], on="qid", how="left")
df = df.merge(deep[["qid","M_deep","p_deep"]], on="qid", how="left")

# 缺的补成中等+0
for col in ["M_mini","M_qwen","M_deep"]:
    df[col] = df[col].fillna("中等M")
for col in ["p_mini","p_qwen","p_deep"]:
    df[col] = df[col].fillna(0.0)

# 把 H_proxy 也变成 M 范式，方便对齐
def h_to_m(h):
    if isinstance(h,str):
        if "简" in h: return "简单M"
        if "中" in h: return "中等M"
        if "难" in h: return "困难M"
    return None

df["H_as_M"] = df["H_proxy"].apply(h_to_m)

os.makedirs("analysis/train_tabs", exist_ok=True)
out = "analysis/train_tabs/race_difficulty_triplet.csv"
df.to_csv(out, index=False, encoding="utf-8")
print("✅ wrote", out, "rows=", len(df))
print(df.head(10).to_string(index=False))
