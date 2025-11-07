import os, re, glob, json, math
import pandas as pd
import numpy as np

ROOT = os.getcwd()
os.makedirs("data", exist_ok=True)

def find_candidate_files():
    pats = ["**/*.csv", "**/*.jsonl"]
    files = []
    for p in pats:
        files.extend(glob.glob(p, recursive=True))
    # 排除我们自己产出的目录
    files = [f for f in files if not any(seg in f for seg in ("/data/", "/tables/", "/figures/", "/.venv", "/venv", "/env/"))]
    return files

COL_ALIASES_QID = ["qid","question_id","id","item_id","q_id"]
COL_ALIASES_CORR = ["is_correct","correct","corr","acc","label_correct","pred_correct","y_true","y"]
COL_ALIASES_CONF = ["conf","confidence","prob","prob_chosen","p_chosen","chosen_prob","proba","probability","p","softmax","score","logprob","logp","chosen_logprob","chosen_logit"]

def pick_col(df, cand, kind):
    for c in df.columns:
        if c.lower() in cand: return c
    # 再尝试包含关系
    low = [c.lower() for c in df.columns]
    for alias in cand:
        for i,name in enumerate(low):
            if alias in name:
                return df.columns[i]
    raise KeyError(f"missing {kind} column; tried {cand}")

def to_conf(s):
    # 如果看起来像 logprob，就 exp
    v = pd.to_numeric(s, errors="coerce")
    if v.min() < 0 and v.max() <= 0:  # 常见 logprob 非正
        v = np.exp(v)
    # clip
    return v.clip(0,1)

def path_has(p, *words):
    pl = p.lower()
    return all(w in pl for w in words)

def detect_dataset(path):
    pl = path.lower()
    if "race" in pl: return "race"
    if "eedi" in pl: return "eedi"
    return None

def infer_model_from_path(path, dataset):
    # 取 dataset 目录后面那一层的名字
    parts = path.split(os.sep)
    try:
        idx = [i for i,s in enumerate(parts) if s.lower()==dataset][0]
        # 向后找第一个非标签、非round的文件夹名
        for j in range(idx+1, max(idx+4, len(parts))):
            if j>=len(parts): break
            name = parts[j].lower()
            if name in ("high","middle","dev","test","train"): continue
            if name.startswith("round") or name.startswith("r"): continue
            if "." in name: continue
            return parts[j]
    except:
        pass
    # 退化：取倒数第二层
    if len(parts)>=2: return parts[-2]
    return "unknown-model"

def infer_round_from_name(path):
    name = os.path.basename(path)
    m = re.search(r"[Rr]ound[_-]?(\d+)", name)
    if not m:
        m = re.search(r"\b[rR](\d+)\b", name)
    if not m:
        m = re.search(r"[_-](\d{1,2})\.", name)
    if m:
        try: return int(m.group(1))
        except: return 1
    # 也尝试从上级目录
    up = os.path.dirname(path)
    m = re.search(r"[Rr]ound[_-]?(\d+)", up)
    if m:
        try: return int(m.group(1))
        except: return 1
    return 1

def load_table(fp):
    if fp.endswith(".csv"):
        return pd.read_csv(fp)
    if fp.endswith(".jsonl"):
        return pd.read_json(fp, lines=True)
    raise ValueError("unsupported ext")

def try_get_human_labels_map():
    # 搜索任何包含 qid + human_label 的 csv
    cands = glob.glob("**/*.csv", recursive=True)
    for f in cands:
        try:
            df = pd.read_csv(f, nrows=3)
        except: 
            continue
        if set(["qid","human_label"]).issubset(set([c.lower() for c in df.columns])):
            df = pd.read_csv(f)
            # 统一列名大小写
            cols = {c:c.lower() for c in df.columns}
            df.rename(columns=cols, inplace=True)
            return df[["qid","human_label"]].copy(), f
    return None, None

def try_get_eedi_diff_map():
    # 优先直接寻找 qid,diff_conf
    cands = glob.glob("**/*.csv", recursive=True)
    for f in cands:
        try:
            df = pd.read_csv(f, nrows=3)
        except: 
            continue
        low = [c.lower() for c in df.columns]
        if "qid" in low and "diff_conf" in low:
            df = pd.read_csv(f)
            cols = {c:c.lower() for c in df.columns}
            df.rename(columns=cols, inplace=True)
            return df[["qid","diff_conf"]].copy(), f
    # 再尝试从学生日志算 diff_conf = 1 - mean(IsCorrect * Confidence/100)
    for f in cands:
        fl = f.lower()
        if "eedi" not in fl: continue
        try:
            df = pd.read_csv(f)
        except: 
            continue
        lowmap = {c:c.lower() for c in df.columns}
        df.rename(columns=lowmap, inplace=True)
        has_qid = "qid" in df.columns
        # 学生正确与信心的常见列名
        cand_corr = [c for c in df.columns if c in ("iscorrect","is_correct","correct","y","label")]
        cand_conf = [c for c in df.columns if "conf" in c]
        if has_qid and cand_corr and cand_conf:
            corr_col = cand_corr[0]
            conf_col = cand_conf[0]
            t = df.groupby("qid", as_index=False).apply(
                lambda g: pd.Series({"diff_conf": 1.0 - np.mean(pd.to_numeric(g[corr_col], errors="coerce").fillna(0).astype(float) * (pd.to_numeric(g[conf_col], errors="coerce").fillna(0).astype(float)/100.0))})
            )
            return t[["qid","diff_conf"]].copy(), f
    return None, None

def main():
    files = find_candidate_files()
    race_rows = []
    eedi_rows = []

    # 预加载映射
    human_map, human_src = try_get_human_labels_map()
    diff_map, diff_src = try_get_eedi_diff_map()

    for fp in files:
        ds = detect_dataset(fp)
        if ds not in ("race","eedi"): 
            continue
        try:
            df = load_table(fp)
        except Exception as e:
            continue
        # 标准化列名映射尝试
        lowmap = {c:c.lower() for c in df.columns}
        df.rename(columns=lowmap, inplace=True)

        try:
            qid_col = pick_col(df, COL_ALIASES_QID, "qid")
        except:
            continue
        # 正确性列
        try:
            corr_col = pick_col(df, COL_ALIASES_CORR, "is_correct")
        except:
            # 如果没有 is_correct，但有 gold 和 pred，可自行比较（很少见，先跳过）
            continue
        # 置信度列
        try:
            conf_col = pick_col(df, COL_ALIASES_CONF, "conf")
        except:
            conf_col = None

        sub = df[[qid_col, corr_col] + ([conf_col] if conf_col else [])].copy()
        sub.rename(columns={qid_col:"qid", corr_col:"is_correct"}, inplace=True)
        if conf_col is not None:
            sub["conf"] = to_conf(sub[conf_col])
        else:
            sub["conf"] = np.nan

        model = infer_model_from_path(fp, ds)
        rnd = infer_round_from_name(fp)

        sub["model"] = model
        if ds=="race":
            sub["round"] = rnd
            # 人类标签合并
            if human_map is not None:
                sub = sub.merge(human_map, on="qid", how="left")
            # fallback: 路径包含 high/middle
            if "human_label" not in sub.columns or sub["human_label"].isna().all():
                sub["human_label"] = np.where(sub["qid"].astype(str).str.contains("high", case=False) | pd.Series([path_has(fp,"high")]*len(sub)), "high", "middle")
            race_rows.append(sub[["qid","model","round","is_correct","conf","human_label"]])
        else:
            eedi_rows.append(sub[["qid","model","is_correct","conf"]])

    if race_rows:
        race_df = pd.concat(race_rows, ignore_index=True)
        race_df["is_correct"] = pd.to_numeric(race_df["is_correct"], errors="coerce").fillna(0).astype(float)
        race_df["conf"] = pd.to_numeric(race_df["conf"], errors="coerce").clip(0,1)
        race_df.to_csv("data/race_runs.csv", index=False)
        print("WROTE data/race_runs.csv", len(race_df))
    else:
        print("WARN: no RACE rows found.")

    if eedi_rows:
        eedi_df = pd.concat(eedi_rows, ignore_index=True)
        eedi_df["is_correct"] = pd.to_numeric(eedi_df["is_correct"], errors="coerce").fillna(0).astype(float)
        eedi_df["conf"] = pd.to_numeric(eedi_df["conf"], errors="coerce")
        eedi_df["conf"] = eedi_df["conf"].where(~eedi_df["conf"].isna(), 0.0).clip(0,1)
        # merge diff_conf
        if diff_map is not None:
            eedi_df = eedi_df.merge(diff_map, on="qid", how="left")
            missing = eedi_df["diff_conf"].isna().sum()
            if missing>0:
                print(f"WARN: diff_conf missing for {missing} rows from {diff_src}.")
        else:
            print("WARN: no diff_conf map found; attempting proxy from is_correct means.")
            proxy = eedi_df.groupby("qid", as_index=False)["is_correct"].mean()
            proxy["diff_conf"] = 1.0 - proxy["is_correct"]
            eedi_df = eedi_df.merge(proxy[["qid","diff_conf"]], on="qid", how="left")
        eedi_df.to_csv("data/eedi_runs.csv", index=False)
        print("WROTE data/eedi_runs.csv", len(eedi_df))
    else:
        print("WARN: no Eedi rows found.")

if __name__ == "__main__":
    main()
