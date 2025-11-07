import os
import json
import pandas as pd

def load_eedi_proxy(path):
    df = pd.read_csv(path)
    # eedi 这份我们知道是 question_id
    df = df.rename(columns={"question_id": "qid"})
    df["question"] = ""      # 等你以后补题面
    df["source"] = "eedi"
    return df[["qid", "question", "H_proxy", "source"]]

def load_race_proxy(path):
    df = pd.read_csv(path)
    cols = df.columns.tolist()

    # 尝试识别 id 列
    id_col = None
    for cand in ["qid", "question_id", "id", "race_id"]:
        if cand in cols:
            id_col = cand
            break
    if id_col is None:
        raise ValueError(f"RACE proxy 没有能当 id 的列，现有列：{cols}")

    # 尝试识别 difficulty 列
    h_col = None
    for cand in ["H_proxy", "difficulty", "human_diff", "human_difficulty"]:
        if cand in cols:
            h_col = cand
            break
    if h_col is None:
        raise ValueError(f"RACE proxy 没有能当难度的列，现有列：{cols}")

    df = df.rename(columns={id_col: "qid", h_col: "H_proxy"})
    df["question"] = ""   # RACE 题干等你后面补
    df["source"] = "race"
    return df[["qid", "question", "H_proxy", "source"]]

def load_synthetic(path):
    rows = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            q = json.loads(line)
            rows.append({
                "qid": q["qid"],
                "question": q.get("stem") or q.get("question") or "",
                "H_proxy": q.get("declared_difficulty", ""),
                "source": "synthetic"
            })
    return pd.DataFrame(rows)

if __name__ == "__main__":
    os.makedirs("baselines/bea24_llm_diff/out", exist_ok=True)

    eedi = load_eedi_proxy("analysis/eedi_proxy_labels.csv")
    race = load_race_proxy("paper_assets/mv-hmda_race/stage2_proxy_labels_race.csv")

    syn_path = "synthetic/gen_questions_200.jsonl"
    if os.path.exists(syn_path):
        syn = load_synthetic(syn_path)
    else:
        syn = pd.DataFrame([], columns=["qid","question","H_proxy","source"])

    all_df = pd.concat([eedi, race, syn], ignore_index=True)

    out_csv = "baselines/bea24_llm_diff/out/all_items_raw.csv"
    all_df.to_csv(out_csv, index=False, encoding="utf-8")
    print("✅ wrote", out_csv, "rows=", len(all_df))
    print("👉 记得后面把 RACE/Eedi 的题面补进来，再跑一次这个脚本。")
