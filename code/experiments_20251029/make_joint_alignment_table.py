import os
import pandas as pd

# 我们已有的两个表
eedi_path = "analysis/train_tabs/eedi_difficulty_triplet.csv"
race_path = "analysis/train_tabs/race_difficulty_triplet.csv"

eedi = pd.read_csv(eedi_path)
race = pd.read_csv(race_path)

# 1) 给数据集打标
eedi["dataset"] = "eedi"
race["dataset"] = "race"

# 2) 为了不要被 RACE 的 75k 完全淹没，我们抽一部分（比如 10k）
#    你要全量就把这行注释掉
if len(race) > 10000:
    race = race.sample(10000, random_state=42).reset_index(drop=True)

# 3) 列对齐说明：
# Eedi: qid, H_proxy, M_gpt4o, gpt4o_p_chosen, M_mini, mini_p_chosen, H_as_M
# RACE: qid, H_proxy, M_mini, p_mini, M_qwen, p_qwen, M_deep, p_deep, H_as_M
# 统一成下面这些列，没有的就填空：
cols = [
    "dataset",
    "qid",
    "H_proxy",
    "H_as_M",
    # 三个“模型位”：slot1, slot2, slot3
    "slot1_label", "slot1_p",
    "slot2_label", "slot2_p",
    "slot3_label", "slot3_p",
]

# eedi 映射：
# slot1 ← mini
# slot2 ← gpt4o
# slot3 ← 空
eedi_norm = pd.DataFrame({
    "dataset": eedi["dataset"],
    "qid": eedi["qid"],
    "H_proxy": eedi["H_proxy"],
    "H_as_M": eedi["H_as_M"],
    "slot1_label": eedi["M_mini"],
    "slot1_p": eedi["mini_p_chosen"],
    "slot2_label": eedi["M_gpt4o"],
    "slot2_p": eedi["gpt4o_p_chosen"],
    "slot3_label": None,
    "slot3_p": 0.0,
})

# race 映射：
# slot1 ← mini
# slot2 ← qwen
# slot3 ← deep
race_norm = pd.DataFrame({
    "dataset": race["dataset"],
    "qid": race["qid"],
    "H_proxy": race["H_proxy"],
    "H_as_M": race["H_as_M"],
    "slot1_label": race["M_mini"],
    "slot1_p": race["p_mini"],
    "slot2_label": race["M_qwen"],
    "slot2_p": race["p_qwen"],
    "slot3_label": race["M_deep"],
    "slot3_p": race["p_deep"],
})

joint = pd.concat([eedi_norm, race_norm], ignore_index=True)

os.makedirs("analysis/joint", exist_ok=True)
out = "analysis/joint/eedi_race_joint.csv"
joint.to_csv(out, index=False, encoding="utf-8")
print(f"✅ wrote {out} rows={len(joint)}")
print(joint.head(10).to_string(index=False))
