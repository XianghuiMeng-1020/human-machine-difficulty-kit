import pandas as pd, os

logs = pd.read_csv("analysis/ednet_flat_with_correct.csv",
                   usecols=["user_id","question_id","correct","timestamp"])
diffs = pd.read_csv("analysis/ednet_proxy_labels_full.csv")

# 映射成 cl4kt 需要的四列：u, q, r, d
diff_map = dict(zip(diffs["item_id"].astype(str), diffs["H_proxy"]))
def h2score(h):
    h = str(h)
    if "简单" in h: return 0.2
    if "中等" in h: return 0.5
    if "困难" in h: return 0.8
    return 0.5

rows = []
for r in logs.itertuples(index=False):
    q = str(r.question_id)
    d = diff_map.get(q, "中等H_proxy")
    rows.append({
        "user_id": r.user_id,
        "item_id": q,
        "correct": r.correct,
        "timestamp": r.timestamp,
        "llm_diff": h2score(d),
    })

out = "baselines/cl4kt_diff/ednet_cl4kt_style.csv"
os.makedirs(os.path.dirname(out), exist_ok=True)
pd.DataFrame(rows).to_csv(out, index=False)
print("✅ wrote", out, "rows=", len(rows))
