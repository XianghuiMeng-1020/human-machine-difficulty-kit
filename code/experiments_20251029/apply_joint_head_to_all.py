import os, numpy as np, pandas as pd

W = np.load("analysis/alignment_head_joint/joint_align_head_W.npy")
D, K = W.shape

LABELS = ["简单M","中等M","困难M"]

def one_hot(label):
    return [1.0 if label == L else 0.0 for L in LABELS]

def encode_row(dataset, slots):
    feats = []
    for i in range(3):
        if i < len(slots) and slots[i] is not None:
            lab, p = slots[i]
            feats.extend(one_hot(lab))
            feats.append(float(p))
        else:
            feats.extend([0.0,0.0,0.0])
            feats.append(0.0)
    is_eedi = 1.0 if dataset=="eedi" else 0.0
    is_race = 1.0 if dataset=="race" else 0.0
    feats.extend([is_eedi, is_race])
    return np.array(feats, dtype=float).reshape(1, -1)

idx2label = {0:"简单M", 1:"中等M", 2:"困难M"}

os.makedirs("analysis/joint_applied", exist_ok=True)

# ---------------- EEDI ----------------
try:
    proxy = pd.read_csv("analysis/eedi_proxy_labels.csv").rename(columns={"question_id":"qid"})
except FileNotFoundError:
    proxy = None

def run_eedi_one(model_csv, out_csv):
    if not os.path.exists(model_csv):
        print("skip (not found):", model_csv)
        return
    df = pd.read_csv(model_csv)
    if proxy is not None:
        df = df.merge(proxy[["qid","H_proxy"]], on="qid", how="left")
    else:
        df["H_proxy"] = None

    aligned = []
    for _, row in df.iterrows():
        lab = row.get("M_tau") or "中等M"
        p = row.get("p_chosen", row.get("p", 0.0))
        x = encode_row("eedi", [(lab, p)])
        logits = x @ W
        pred = int(logits.argmax(axis=1)[0])
        aligned.append(idx2label[pred])
    df["aligned_h_like"] = aligned

    # eval
    if "H_proxy" in df.columns:
        def h2m(h):
            if pd.isna(h): return None
            h = str(h)
            if "简" in h: return "简单M"
            if "中" in h: return "中等M"
            if "难" in h: return "困难M"
            return None
        df["H_as_M"] = df["H_proxy"].apply(h2m)
        mask = df["H_as_M"].notna()
        if mask.any():
            base = (df.loc[mask, "M_tau"] == df.loc[mask, "H_as_M"]).mean()
            head = (df.loc[mask, "aligned_h_like"] == df.loc[mask, "H_as_M"]).mean()
            print(f"[Eedi] {model_csv}  N={mask.sum()}  base={base:.4f}  joint-head={head:.4f}")

    df.to_csv(out_csv, index=False, encoding="utf-8")
    print("✅ wrote", out_csv)

run_eedi_one("analysis/eedi_gpt4o_tau08_model_tags.csv",
             "analysis/joint_applied/eedi_gpt4o_tau08_joint.csv")
run_eedi_one("analysis/eedi_gpt4omini_tau08_model_tags.csv",
             "analysis/joint_applied/eedi_gpt4omini_tau08_joint.csv")

# ---------------- RACE ----------------
race_proxy_path = "paper_assets/mv-hmda_race/stage2_proxy_labels_race.csv"
race_proxy = None
if os.path.exists(race_proxy_path):
    tmp = pd.read_csv(race_proxy_path)
    # 尝试识别qid列
    cand = None
    for c in ["qid", "question_id", "id", "q_id"]:
        if c in tmp.columns:
            cand = c
            break
    if cand is None:
        raise RuntimeError(f"race proxy 没有能认出的题号列: {tmp.columns.tolist()}")
    tmp = tmp.rename(columns={cand: "qid"})
    race_proxy = tmp

def run_race_one(model_csv, out_csv):
    if not os.path.exists(model_csv):
        print("skip (not found):", model_csv)
        return
    df = pd.read_csv(model_csv)

    if race_proxy is not None:
        df = df.merge(race_proxy[["qid","H_proxy"]], on="qid", how="left")
    else:
        df["H_proxy"] = None

    aligned = []
    for _, row in df.iterrows():
        lab = row.get("M_tau") or "中等M"
        p = row.get("p_chosen", row.get("p", 1.0))
        x = encode_row("race", [(lab, p)])
        logits = x @ W
        pred = int(logits.argmax(axis=1)[0])
        aligned.append(idx2label[pred])
    df["aligned_h_like"] = aligned

    if "H_proxy" in df.columns:
        def h2m(h):
            if pd.isna(h): return None
            h = str(h)
            if "简" in h: return "简单M"
            if "中" in h: return "中等M"
            if "难" in h: return "困难M"
            return None
        df["H_as_M"] = df["H_proxy"].apply(h2m)
        mask = df["H_as_M"].notna()
        if mask.any():
            base = (df.loc[mask, "M_tau"] == df.loc[mask, "H_as_M"]).mean()
            head = (df.loc[mask, "aligned_h_like"] == df.loc[mask, "H_as_M"]).mean()
            print(f"[RACE] {model_csv}  N={mask.sum()}  base={base:.4f}  joint-head={head:.4f}")

    df.to_csv(out_csv, index=False, encoding="utf-8")
    print("✅ wrote", out_csv)

run_race_one("paper_assets/mv-hmda_race/stage2_model_tags_race_gpt4omini_tau08.csv",
             "analysis/joint_applied/race_gpt4omini_tau08_joint.csv")
run_race_one("paper_assets/mv-hmda_race/stage2_model_tags_race_qwen3next80b_tau08.csv",
             "analysis/joint_applied/race_qwen3next80b_tau08_joint.csv")
run_race_one("paper_assets/mv-hmda_race/stage2_model_tags_race_deepseekv3_tau08.csv",
             "analysis/joint_applied/race_deepseekv3_tau08_joint.csv")

print("🎯 joint head applied.")
