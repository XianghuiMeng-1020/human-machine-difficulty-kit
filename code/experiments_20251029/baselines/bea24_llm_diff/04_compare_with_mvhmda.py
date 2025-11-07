import os
import pandas as pd

BASE = "baselines/bea24_llm_diff/out"
pred_path = os.path.join(BASE, "bea24_stub_preds.csv")
pred = pd.read_csv(pred_path)

# ------------------------------
# 1) Eedi gold
# ------------------------------
eedi_gold = pd.read_csv("analysis/eedi_proxy_labels.csv").rename(
    columns={"question_id": "qid"}
)
eedi_gold["qid"] = eedi_gold["qid"].astype(str)
eedi_gold = eedi_gold[["qid", "H_proxy"]]

# ------------------------------
# 2) RACE gold  —— 列名自适应
# ------------------------------
race_gold_raw = pd.read_csv("paper_assets/mv-hmda_race/stage2_proxy_labels_race.csv")

def find_id_col(cols):
    # 尽量从最可能的开始
    cand_order = ["qid", "question_id", "id", "item_id"]
    lower = {c.lower(): c for c in cols}
    for c in cand_order:
        if c in lower:
            return lower[c]
    # 还不行就找里面带 'id' 的
    for c in cols:
        if "id" in c.lower():
            return c
    raise ValueError("⚠️ cannot find id-like column in RACE proxy file")

race_id_col = find_id_col(race_gold_raw.columns)
race_gold = race_gold_raw.rename(columns={race_id_col: "qid"})
race_gold["qid"] = race_gold["qid"].astype(str)
# 同样要有 H_proxy，若没有就报清楚
if "H_proxy" not in race_gold.columns:
    raise ValueError("⚠️ RACE proxy file has no 'H_proxy' column. Check paper_assets/mv-hmda_race/stage2_proxy_labels_race.csv")
race_gold = race_gold[["qid", "H_proxy"]]

rows = []

# =========================================================
# EEDI part
# =========================================================
eedi_pred = pred[pred["source"] == "eedi"].copy()
if len(eedi_pred):
    eedi_pred["qid"] = eedi_pred["qid"].astype(str)
    eedi_pred = eedi_pred.merge(eedi_gold, on="qid", how="left", suffixes=("", "_gold"))

    # baseline 预测字段
    if "pred_h_like" in eedi_pred.columns:
        eedi_pred["pred"] = eedi_pred["pred_h_like"]
    else:
        eedi_pred["pred"] = eedi_pred["pred_label"]

    mask = eedi_pred["H_proxy"].notna()
    acc = (eedi_pred.loc[mask, "pred"] == eedi_pred.loc[mask, "H_proxy"]).mean() if mask.any() else float("nan")

    rows.append({
        "dataset": "Eedi",
        "baseline": "BEA24-LLM-diff (stub feats)",
        "acc_or_align": acc,
        "notes": f"N={mask.sum()}"
    })
else:
    rows.append({
        "dataset": "Eedi",
        "baseline": "BEA24-LLM-diff (stub feats)",
        "acc_or_align": float("nan"),
        "notes": "no eedi rows in preds"
    })

# 再把你的方法塞进去做横向
rows.append({
    "dataset": "Eedi",
    "baseline": "MV-HMDA (autonorm) gpt4o",
    "acc_or_align": 0.70,
    "notes": "analysis/eedi_proxy_x_gpt4o_true_autonorm.csv"
})
rows.append({
    "dataset": "Eedi",
    "baseline": "MV-HMDA (autonorm) mini",
    "acc_or_align": 0.80,
    "notes": "analysis/eedi_proxy_x_gpt4omini_true_autonorm.csv"
})

# =========================================================
# RACE part
# =========================================================
race_pred = pred[pred["source"] == "race"].copy()
if len(race_pred):
    race_pred["qid"] = race_pred["qid"].astype(str)
    race_pred = race_pred.merge(race_gold, on="qid", how="left", suffixes=("", "_gold"))

    # baseline 预测字段
    if "pred_h_like_race" in race_pred.columns:
        race_pred["pred"] = race_pred["pred_h_like_race"]
    elif "pred_h_like" in race_pred.columns:
        race_pred["pred"] = race_pred["pred_h_like"]
    else:
        race_pred["pred"] = race_pred["pred_label"]

    mask = race_pred["H_proxy"].notna()
    acc = (race_pred.loc[mask, "pred"] == race_pred.loc[mask, "H_proxy"]).mean() if mask.any() else float("nan")

    rows.append({
        "dataset": "RACE",
        "baseline": "BEA24-LLM-diff (stub feats)",
        "acc_or_align": acc,
        "notes": f"N={mask.sum()} (id_col={race_id_col})"
    })
else:
    rows.append({
        "dataset": "RACE",
        "baseline": "BEA24-LLM-diff (stub feats)",
        "acc_or_align": float("nan"),
        "notes": "no race rows in preds"
    })

# 你的 RACE 上限
rows.append({
    "dataset": "RACE",
    "baseline": "MV-HMDA proxy-5runs (mini)",
    "acc_or_align": 0.8617,
    "notes": "paper_assets/mv-hmda_race/... mini"
})
rows.append({
    "dataset": "RACE",
    "baseline": "MV-HMDA proxy-5runs (qwen)",
    "acc_or_align": 0.8640,
    "notes": "paper_assets/mv-hmda_race/... qwen"
})
rows.append({
    "dataset": "RACE",
    "baseline": "MV-HMDA proxy-5runs (deepseek)",
    "acc_or_align": 0.8597,
    "notes": "paper_assets/mv-hmda_race/... deepseek"
})

out_csv = os.path.join(BASE, "bea24_vs_mvhmda.csv")
pd.DataFrame(rows).to_csv(out_csv, index=False, encoding="utf-8")

print("=== SUMMARY ===")
for r in rows:
    print(r)
print(f"✅ wrote {out_csv}")
