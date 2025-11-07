import os, pandas as pd

rows = []

# ========== 小工具 ==========
def h_to_m(h):
    if not isinstance(h, str):
        return None
    if "简" in h or "easy" in h.lower():
        return "简单M"
    if "中" in h or "mid" in h.lower():
        return "中等M"
    if "难" in h or "hard" in h.lower():
        return "困难M"
    return None

def pick_head_col(df):
    # 不同脚本写的列名有点不一样，这里兜一下
    for col in ["aligned_from_joint", "pred_label", "aligned_h_like"]:
        if col in df.columns:
            return col
    return None

# ========== 1) Eedi ==========
proxy_path = "analysis/eedi_proxy_labels.csv"
if os.path.exists(proxy_path):
    proxy = pd.read_csv(proxy_path).rename(columns={"question_id":"qid"})
else:
    proxy = None

eedi_models = [
    ("Eedi", "gpt4o",
     "analysis/eedi_gpt4o_tau08_model_tags.csv",
     "analysis/joint_applied/eedi_gpt4o_tau08_joint.csv"),
    ("Eedi", "gpt4o-mini",
     "analysis/eedi_gpt4omini_tau08_model_tags.csv",
     "analysis/joint_applied/eedi_gpt4omini_tau08_joint.csv"),
]

for ds, model, base_csv, head_csv in eedi_models:
    if not os.path.exists(base_csv):
        continue
    base = pd.read_csv(base_csv)
    if proxy is not None:
        base = base.merge(proxy[["qid","H_proxy"]], on="qid", how="left")
    # 原始对齐
    if "M_tau" in base.columns and "H_proxy" in base.columns:
        h_as_m = base["H_proxy"].apply(h_to_m)
        base_align = (base["M_tau"] == h_as_m).mean()
    else:
        base_align = float("nan")

    # joint-head 后
    head_align = float("nan")
    if os.path.exists(head_csv):
        head = pd.read_csv(head_csv)
        if proxy is not None:
            head = head.merge(proxy[["qid","H_proxy"]], on="qid", how="left")
        col = pick_head_col(head)
        if col and "H_proxy" in head.columns:
            h_as_m = head["H_proxy"].apply(h_to_m)
            head_align = (head[col] == h_as_m).mean()

    rows.append({
        "dataset": ds,
        "model": model,
        "variant": "base",
        "alignment": round(base_align, 4),
        "n": len(base),
    })
    rows.append({
        "dataset": ds,
        "model": model,
        "variant": "joint-head",
        "alignment": round(head_align, 4),
        "n": len(base),
    })

# ========== 2) RACE ==========
race_proxy_path = "paper_assets/mv-hmda_race/stage2_proxy_labels_race.csv"
if os.path.exists(race_proxy_path):
    race_proxy = pd.read_csv(race_proxy_path)
    # 你之前的 race proxy 应该有 qid，没有的话就 rename 一下
    if "question_id" in race_proxy.columns and "qid" not in race_proxy.columns:
        race_proxy = race_proxy.rename(columns={"question_id":"qid"})
else:
    race_proxy = None

race_models = [
    ("RACE", "gpt4omini",
     "paper_assets/mv-hmda_race/stage2_model_tags_race_gpt4omini_tau08.csv",
     "analysis/joint_applied/race_gpt4omini_tau08_joint.csv"),
    ("RACE", "qwen3next80b",
     "paper_assets/mv-hmda_race/stage2_model_tags_race_qwen3next80b_tau08.csv",
     "analysis/joint_applied/race_qwen3next80b_tau08_joint.csv"),
    ("RACE", "deepseekv3",
     "paper_assets/mv-hmda_race/stage2_model_tags_race_deepseekv3_tau08.csv",
     "analysis/joint_applied/race_deepseekv3_tau08_joint.csv"),
]

for ds, model, base_csv, head_csv in race_models:
    if not os.path.exists(base_csv):
        continue
    base = pd.read_csv(base_csv)
    if race_proxy is not None:
        base = base.merge(race_proxy[["qid","H_proxy"]], on="qid", how="left")
    # 原始对齐
    if "M_tau" in base.columns and "H_proxy" in base.columns:
        h_as_m = base["H_proxy"].apply(h_to_m)
        base_align = (base["M_tau"] == h_as_m).mean()
    else:
        base_align = float("nan")

    # joint-head 后
    head_align = float("nan")
    if os.path.exists(head_csv):
        head = pd.read_csv(head_csv)
        if race_proxy is not None:
            head = head.merge(race_proxy[["qid","H_proxy"]], on="qid", how="left")
        col = pick_head_col(head)
        if col and "H_proxy" in head.columns:
            h_as_m = head["H_proxy"].apply(h_to_m)
            head_align = (head[col] == h_as_m).mean()

    rows.append({
        "dataset": ds,
        "model": model,
        "variant": "base",
        "alignment": round(base_align, 4),
        "n": len(base),
    })
    rows.append({
        "dataset": ds,
        "model": model,
        "variant": "joint-head",
        "alignment": round(head_align, 4),
        "n": len(base),
    })

# ========== 3) EdNet (我们就拿 coverage 那四张表) ==========
ednet_variants = [
    ("EdNet-KT1", "slice-u200",  "analysis/ednet_labels_u200.csv"),
    ("EdNet-KT1", "slice-u200b", "analysis/ednet_labels_u200b.csv"),
    ("EdNet-KT1", "slice-u500",  "analysis/ednet_labels_u500.csv"),
    ("EdNet-KT1", "slice-u1000", "analysis/ednet_labels_u1000.csv"),
]
for ds, var, csv_path in ednet_variants:
    if not os.path.exists(csv_path):
        continue
    df = pd.read_csv(csv_path)
    if "H_proxy" in df.columns:
        # 这里的“alignment”我们就用 “不是低曝光”的比例，和你前面 global 表一致
        aligned = (df["H_proxy"] != "低曝光").mean()
        rows.append({
            "dataset": ds,
            "model": "-",
            "variant": var,
            "alignment": round(aligned, 4),
            "n": len(df),
        })

# ========== 写出 ==========
os.makedirs("analysis/global", exist_ok=True)
out = "analysis/global/alignment_runs_all.csv"
pd.DataFrame(rows).to_csv(out, index=False, encoding="utf-8")
print("✅ wrote", out, "rows=", len(rows))
