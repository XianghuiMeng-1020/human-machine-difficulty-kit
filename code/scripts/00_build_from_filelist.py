
import os, glob, pandas as pd, numpy as np
from pathlib import Path

def _read_perq_csv(p):
    df = pd.read_csv(p)
    df["qid"] = df["qid"].astype(str)
    # 置信度优先 prob_* 最大值；否则保留 conf
    prob_cols = [c for c in df.columns if str(c).startswith("prob_")]
    if prob_cols:
        df["conf"] = df[prob_cols].apply(pd.to_numeric, errors="coerce").max(axis=1)
    else:
        df["conf"] = pd.to_numeric(df.get("conf"), errors="coerce")
    df["is_correct"] = pd.to_numeric(df.get("is_correct"), errors="coerce")
    return df[["qid","is_correct","conf"]].copy()

def build_race(race_glob="experiments_20251029/results/race/*/round1/per_question.csv",
               out_runs="data/race_runs.csv",
               map_path="data/race/human_diff_map.csv"):
    Path("data/race").mkdir(parents=True, exist_ok=True)
    rows=[]
    for p in sorted(glob.glob(race_glob)):
        # 模型名取目录名（.../race/<model>/round1/per_question.csv）
        parts = Path(p).parts
        try:
            model = parts[parts.index("race")+1]
        except Exception:
            model = Path(p).parent.parent.name  # 兜底
        df = _read_perq_csv(p)
        df["model"] = model
        rows.append(df)
    if not rows:
        print("WARN: no RACE per_question.csv found; writing empty runs.")
        pd.DataFrame(columns=["qid","model","is_correct","conf"]).to_csv(out_runs, index=False)
        return

    runs = pd.concat(rows, ignore_index=True)
    # 合并人类难度映射
    if os.path.exists(map_path):
        mp = pd.read_csv(map_path, dtype=str)
        mp = mp.rename(columns={c: c.strip() for c in mp.columns})
        mp["qid"] = mp["qid"].astype(str).str.strip()
        mp["human_label"] = mp["human_label"].astype(str).str.strip().str.lower()
        mp["diff_conf"]   = pd.to_numeric(mp.get("diff_conf"), errors="coerce")
        runs = runs.merge(mp[["qid","human_label","diff_conf"]], on="qid", how="left")
        print(f"[OK] merged human_diff_map -> {out_runs}  coverage(diff_conf)=", 
              float(runs["diff_conf"].notna().mean()))
    else:
        print("WARN: no diff_conf map; used proxy 1-mean(IsCorrect)")

    runs.to_csv(out_runs, index=False)
    print(f"WROTE {out_runs} {len(runs)} rows models=", sorted(runs["model"].dropna().unique().tolist()))

def build_eedi(eedi_glob="experiments_20251029/results/eedi/*/per_question.csv",
               out_runs="data/eedi_runs.csv"):
    rows=[]
    for p in sorted(glob.glob(eedi_glob)):
        df = pd.read_csv(p)
        if "qid" not in df: 
            continue
        df["qid"] = df["qid"].astype(str)
        df["model"] = Path(p).parent.parent.name
        df["is_correct"] = pd.to_numeric(df.get("is_correct"), errors="coerce")
        df["conf"] = pd.to_numeric(df.get("conf"), errors="coerce")
        rows.append(df[["qid","model","is_correct","conf"]])
    if rows:
        out = pd.concat(rows, ignore_index=True)
        out.to_csv(out_runs, index=False)
        print(f"WROTE {out_runs} {len(out)}")
    else:
        pd.DataFrame(columns=["qid","model","is_correct","conf"]).to_csv(out_runs, index=False)
        print("WARN: no EDNet rows")

if __name__ == "__main__":
    Path("data").mkdir(exist_ok=True)
    build_race()
    build_eedi()
    print("DONE.")
