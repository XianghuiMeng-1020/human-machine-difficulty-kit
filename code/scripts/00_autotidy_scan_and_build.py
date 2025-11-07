import os, re, glob
import pandas as pd
import numpy as np

os.makedirs("data", exist_ok=True)

def find_files():
    files=[]
    for p in ("**/race/**/*.csv","**/race/**/*.jsonl","**/eedi/**/*.csv","**/eedi/**/*.jsonl","**/ednet/**/*.csv","**/ednet/**/*.jsonl","**/kt1/**/*.csv","**/kt1/**/*.jsonl"):
        files+=glob.glob(p, recursive=True)
    EXCLUDE=('/.git/','/.venv/','/venv/','/env/','/node_modules/','/.ipynb_checkpoints/','/Library/')
    return [f for f in files if not any(seg in f for seg in EXCLUDE)]

ALIASES_QID=["qid","question_id","id","item_id","q_id","problem_id","item"]
ALIASES_CORR=["is_correct","correct","corr","acc","label_correct","pred_correct","y_true","y","answer","answered_correctly"]
ALIASES_CONF=["conf","confidence","prob","prob_chosen","p_chosen","chosen_prob","proba","probability","p","softmax","score","logprob","logp","chosen_logprob","chosen_logit"]

def pick_col(df,cands):
    low=[c.lower() for c in df.columns]
    for i,c in enumerate(low):
        if c in cands: return df.columns[i]
    for alias in cands:
        for i,c in enumerate(low):
            if alias in c: return df.columns[i]
    raise KeyError

def to_conf(s):
    v=pd.to_numeric(s, errors="coerce")
    if v.min()<0 and v.max()<=0: v=np.exp(v)
    return v.clip(0,1)

def detect_ds(path):
    pl=path.lower()
    if "/race/" in pl: return "race"
    if "/eedi/" in pl: return "eedi"
    if "/ednet/" in pl or "/kt1/" in pl: return "ednet"
    return None

def infer_model(path, ds):
    parts=path.split(os.sep)
    idx=None
    for i,s in enumerate(parts):
        if s.lower()==ds or (ds=="ednet" and s.lower() in ("ednet","kt1")):
            idx=i; break
    if idx is None: return parts[-2] if len(parts)>=2 else "unknown"
    for j in range(idx+1, min(idx+5,len(parts))):
        name=parts[j].lower()
        if name in ("high","middle","dev","test","train","logs"): continue
        if name.startswith("round") or name.startswith("r"): continue
        if "." in name: continue
        return parts[j]
    return parts[-2] if len(parts)>=2 else "unknown"

def infer_round(path):
    name=os.path.basename(path)
    m=re.search(r"[Rr]ound[_-]?(\d+)",name) or re.search(r"\b[rR](\d+)\b",name) or re.search(r"[_-](\d{1,2})\.",name)
    if m:
        try: return int(m.group(1))
        except: return 1
    up=os.path.dirname(path)
    m=re.search(r"[Rr]ound[_-]?(\d+)",up)
    if m:
        try: return int(m.group(1))
        except: return 1
    return 1

def try_map_human(files):
    for f in files:
        if not f.lower().endswith(".csv"): continue
        try: df=pd.read_csv(f, nrows=3)
        except: continue
        low=[c.lower() for c in df.columns]
        if "qid" in low and "human_label" in low:
            df=pd.read_csv(f); df.columns=[c.lower() for c in df.columns]
            df["qid"]=df["qid"].astype(str)
            return df[["qid","human_label"]].copy()
    return None

def try_diff_conf(files):
    for f in files:
        if not f.lower().endswith(".csv"): continue
        try: df=pd.read_csv(f, nrows=3)
        except: continue
        low=[c.lower() for c in df.columns]
        if "qid" in low and "diff_conf" in low:
            df=pd.read_csv(f); df.columns=[c.lower() for c in df.columns]
            df["qid"]=df["qid"].astype(str)
            return df[["qid","diff_conf"]].copy()
    return None

files=find_files()
human_map=try_map_human(files)
diff_map=try_diff_conf(files)

race_rows=[]; eedi_rows=[]; ednet_rows=[]; ednet_human_rows=[]

for fp in files:
    ds=detect_ds(fp)
    if ds is None: continue
    try:
        df=pd.read_json(fp, lines=True) if fp.endswith(".jsonl") else pd.read_csv(fp)
    except:
        continue
    df.columns=[c.lower() for c in df.columns]
    # 针对 ednet 学生日志，先记录人类侧
    if ds=="ednet":
        if "answered_correctly" in df.columns or "is_correct" in df.columns or "correct" in df.columns:
            try:
                qid=pick_col(df,ALIASES_QID)
                corr=pick_col(df,ALIASES_CORR)
            except:
                qid=None; corr=None
            if qid and corr:
                tmp=df[[qid,corr]].copy()
                tmp.rename(columns={qid:"qid",corr:"is_correct"}, inplace=True)
                tmp["qid"]=tmp["qid"].astype(str)
                tmp["is_correct"]=pd.to_numeric(tmp["is_correct"], errors="coerce").fillna(0).astype(float)
                ednet_human_rows.append(tmp)
    # 模型端
    try: qid=pick_col(df,ALIASES_QID)
    except: continue
    try: corr=pick_col(df,ALIASES_CORR)
    except: continue
    conf=None
    try: conf=pick_col(df,ALIASES_CONF)
    except: pass
    sub=df[[qid,corr]+([conf] if conf else [])].copy()
    sub.rename(columns={qid:"qid",corr:"is_correct"}, inplace=True)
    sub["qid"]=sub["qid"].astype(str)
    sub["is_correct"]=pd.to_numeric(sub["is_correct"], errors="coerce").fillna(0).astype(float)
    sub["conf"]=to_conf(sub[conf]) if conf else 0.0
    model=infer_model(fp, ds)
    if ds=="race":
        sub["model"]=model; sub["round"]=infer_round(fp)
        if human_map is not None:
            sub=sub.merge(human_map, on="qid", how="left")
        if "human_label" not in sub.columns or sub["human_label"].isna().all():
            flag=("high" in fp.lower()); sub["human_label"]=np.where(flag,"high","middle")
        race_rows.append(sub[["qid","model","round","is_correct","conf","human_label"]])
    elif ds=="eedi":
        sub["model"]=model
        eedi_rows.append(sub[["qid","model","is_correct","conf"]])
    else:
        sub["model"]=model
        ednet_rows.append(sub[["qid","model","is_correct","conf"]])

os.makedirs("data", exist_ok=True)

if race_rows:
    race=pd.concat(race_rows, ignore_index=True)
    race.to_csv("data/race_runs.csv", index=False)
    print("WROTE data/race_runs.csv", len(race))
else:
    print("WARN: no RACE rows found.")

if eedi_rows:
    eedi=pd.concat(eedi_rows, ignore_index=True)
    if diff_map is None:
        proxy=eedi.groupby("qid", as_index=False)["is_correct"].mean()
        proxy["diff_conf"]=1.0 - proxy["is_correct"]
        eedi=eedi.merge(proxy[["qid","diff_conf"]], on="qid", how="left")
        print("WARN: no diff_conf map; used proxy 1-mean(IsCorrect) for Eedi.")
    else:
        eedi=eedi.merge(diff_map, on="qid", how="left")
    eedi.to_csv("data/eedi_runs.csv", index=False)
    print("WROTE data/eedi_runs.csv", len(eedi))
else:
    print("WARN: no Eedi rows found.")

if ednet_rows:
    ednet=pd.concat(ednet_rows, ignore_index=True)
    if ednet_human_rows:
        hh=pd.concat(ednet_human_rows, ignore_index=True)
        hum=hh.groupby("qid", as_index=False)["is_correct"].mean()
        hum.rename(columns={"is_correct":"human_acc"}, inplace=True)
        ednet=ednet.merge(hum, on="qid", how="left")
    else:
        hum=ednet.groupby("qid", as_index=False)["is_correct"].mean().rename(columns={"is_correct":"human_acc"})
        ednet=ednet.merge(hum, on="qid", how="left")
    ednet["human_err"]=1.0 - ednet["human_acc"]
    ednet.to_csv("data/ednet_runs.csv", index=False)
    print("WROTE data/ednet_runs.csv", len(ednet))
else:
    print("WARN: no EDNet rows found.")
