import pandas as pd
import os

# 1) 读进来
proxy = pd.read_csv("analysis/eedi_proxy_labels.csv")  # has question_id
g4o   = pd.read_csv("analysis/eedi_gpt4o_tau08_model_tags.csv")  # has qid
mini  = pd.read_csv("analysis/eedi_gpt4omini_tau08_model_tags.csv")  # has qid

def norm_qid(df, candidates=("qid", "question_id", "item_id", "\ufeffqid")):
    cols = list(df.columns)
    for c in candidates:
        if c in cols:
            return df.rename(columns={c: "qid"})
    # 实在没有就用第一列
    return df.rename(columns={cols[0]: "qid"})

proxy = norm_qid(proxy)
g4o   = norm_qid(g4o)
mini  = norm_qid(mini)

# 2) 模型打的标签列名统一一下
g4o = g4o.rename(columns={"M_tau": "M_gpt4o"})
mini = mini.rename(columns={"M_tau": "M_mini"})

# 3) 左连接到 proxy 上
df = proxy.merge(g4o[["qid", "M_gpt4o", "p_chosen", "correct"]].rename(
    columns={"p_chosen": "gpt4o_p_chosen", "correct": "gpt4o_correct_flag"}
), on="qid", how="left")

df = df.merge(mini[["qid", "M_mini", "p_chosen", "correct"]].rename(
    columns={"p_chosen": "mini_p_chosen", "correct": "mini_correct_flag"}
), on="qid", how="left")

# 4) 把 proxy 的中文难度映射成模型三档，方便后面训练
def h_to_m(label: str):
    if not isinstance(label, str):
        return None
    label = label.strip()
    if "易" in label or "simple" in label.lower() or "easy" in label.lower():
        return "简单M"
    if "中" in label or "mid" in label.lower() or "medium" in label.lower():
        return "中等M"
    if "难" in label or "hard" in label.lower():
        return "困难M"
    return None

if "H_proxy" in df.columns:
    df["H_as_M"] = df["H_proxy"].apply(h_to_m)
else:
    df["H_as_M"] = None

# 5) 存
os.makedirs("analysis/train_tabs", exist_ok=True)
out = "analysis/train_tabs/eedi_difficulty_triplet.csv"
df.to_csv(out, index=False, encoding="utf-8")
print("✅ wrote", out, "rows=", len(df))
print(df.head(10).to_string(index=False))
