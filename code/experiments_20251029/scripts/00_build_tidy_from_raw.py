import argparse, glob, os, re, json
import pandas as pd

def load_one(path, fmt, qid_col, corr_col, conf_col):
    if fmt=="csv":
        df = pd.read_csv(path)
    elif fmt=="jsonl":
        df = pd.read_json(path, lines=True)
    else:
        raise ValueError("fmt must be csv|jsonl")
    need = []
    for c in [qid_col, corr_col, conf_col]:
        if c not in df.columns: need.append(c)
    if need:
        raise ValueError(f"{path} missing columns: {need}")
    return df[[qid_col, corr_col, conf_col]].rename(columns={
        qid_col:"qid", corr_col:"is_correct", conf_col:"conf"
    })

def infer_with_regex(path, regex, group):
    m = re.search(regex, path)
    if not m: return None
    g = m.group(group)
    return g

def build_race(args):
    rows = []
    files = sorted([f for pat in args.race_globs for f in glob.glob(pat)])
    if not files:
        print("RACE: no files matched. Skipping.")
        return None
    for fp in files:
        try:
            df = load_one(fp, args.race_fmt, args.race_qid, args.race_correct, args.race_conf)
        except Exception as e:
            print(f"[RACE] skip {fp}: {e}")
            continue
        # 模型名
        model = args.race_model_static or infer_with_regex(fp, args.race_model_re, args.race_model_group)
        # 轮次
        rstr = infer_with_regex(fp, args.race_round_re, args.race_round_group) if args.race_round_re else None
        try:
            rnd = int(rstr) if rstr is not None else 1
        except:
            rnd = 1
        df["model"] = model
        df["round"] = rnd
        df["src_path"] = fp
        rows.append(df)
    if not rows:
        print("RACE: no usable files.")
        return None
    out = pd.concat(rows, ignore_index=True)
    # 人类标签
    if args.race_human_map and os.path.exists(args.race_human_map):
        h = pd.read_csv(args.race_human_map)
        if not set(["qid","human_label"]).issubset(h.columns):
            raise ValueError("race_human_map must have columns: qid,human_label")
        out = out.merge(h[["qid","human_label"]], on="qid", how="left")
    else:
        # 尝试从路径推断（若文件路径中含 middle/high）
        out["human_label"] = out["src_path"].str.contains(r"high", case=False).map({True:"high", False:"middle"})
    # 类型清洗
    out["is_correct"] = out["is_correct"].astype(float)
    out["conf"] = out["conf"].astype(float).clip(0,1)
    if args.dryrun:
        print("RACE preview:", out.head(3).to_dict(orient="records"))
        print("RACE counts:", out.groupby(["model","round"]).size().head(10))
    else:
        os.makedirs("data", exist_ok=True)
        out[["qid","model","round","is_correct","conf","human_label"]].to_csv("data/race_runs.csv", index=False)
        print("Wrote data/race_runs.csv", len(out))
    return out

def build_eedi(args):
    rows = []
    files = sorted([f for pat in args.eedi_globs for f in glob.glob(pat)])
    if not files:
        print("Eedi: no files matched. Skipping.")
        return None
    for fp in files:
        try:
            df = load_one(fp, args.eedi_fmt, args.eedi_qid, args.eedi_correct, args.eedi_conf)
        except Exception as e:
            print(f"[Eedi] skip {fp}: {e}")
            continue
        model = args.eedi_model_static or infer_with_regex(fp, args.eedi_model_re, args.eedi_model_group)
        df["model"] = model
        df["src_path"] = fp
        rows.append(df)
    if not rows:
        print("Eedi: no usable files.")
        return None
    out = pd.concat(rows, ignore_index=True)
    # 合并 diff_conf
    if not args.eedi_diff_map or not os.path.exists(args.eedi_diff_map):
        raise ValueError("Need --eedi_diff_map CSV with columns qid,diff_conf")
    diff = pd.read_csv(args.eedi_diff_map)
    if not set(["qid","diff_conf"]).issubset(diff.columns):
        raise ValueError("--eedi_diff_map must have columns: qid,diff_conf")
    out = out.merge(diff[["qid","diff_conf"]], on="qid", how="left")
    out["is_correct"] = out["is_correct"].astype(float)
    out["conf"] = out["conf"].astype(float).clip(0,1)
    if args.dryrun:
        print("Eedi preview:", out.head(3).to_dict(orient="records"))
        print("Eedi counts:", out.groupby(["model"]).size())
    else:
        os.makedirs("data", exist_ok=True)
        out[["qid","model","is_correct","conf","diff_conf"]].to_csv("data/eedi_runs.csv", index=False)
        print("Wrote data/eedi_runs.csv", len(out))
    return out

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dryrun", action="store_true", help="only print preview, do not write data/*.csv")

    # ---------- RACE ----------
    ap.add_argument("--race-glob", dest="race_globs", action="append", default=[], help="glob(s) for RACE model outputs")
    ap.add_argument("--race-fmt", dest="race_fmt", default="csv", choices=["csv","jsonl"])
    ap.add_argument("--race-qid", dest="race_qid", default="qid")
    ap.add_argument("--race-correct", dest="race_correct", default="is_correct")
    ap.add_argument("--race-conf", dest="race_conf", default="conf")
    ap.add_argument("--race-model-re", dest="race_model_re", default=r"/([^/]+)/[^/]*$")
    ap.add_argument("--race-model-group", dest="race_model_group", type=int, default=1)
    ap.add_argument("--race-round-re", dest="race_round_re", default=r"[Rr]ound[_-]?(\d+)|r(\d+)")
    ap.add_argument("--race-round-group", dest="race_round_group", type=int, default=1)
    ap.add_argument("--race-model-static", dest="race_model_static", default=None)
    ap.add_argument("--race-human-map", dest="race_human_map", default=None, help="CSV with qid,human_label")

    # ---------- Eedi ----------
    ap.add_argument("--eedi-glob", dest="eedi_globs", action="append", default=[], help="glob(s) for Eedi model outputs")
    ap.add_argument("--eedi-fmt", dest="eedi_fmt", default="csv", choices=["csv","jsonl"])
    ap.add_argument("--eedi-qid", dest="eedi_qid", default="qid")
    ap.add_argument("--eedi-correct", dest="eedi_correct", default="is_correct")
    ap.add_argument("--eedi-conf", dest="eedi_conf", default="conf")
    ap.add_argument("--eedi-model-re", dest="eedi_model_re", default=r"/([^/]+)/[^/]*$")
    ap.add_argument("--eedi-model-group", dest="eedi_model_group", type=int, default=1)
    ap.add_argument("--eedi-model-static", dest="eedi_model_static", default=None)
    ap.add_argument("--eedi-diff-map", dest="eedi_diff_map", default=None, help="CSV with qid,diff_conf")

    args = ap.parse_args()

    if args.race_globs:
        build_race(args)
    if args.eedi_globs:
        build_eedi(args)
    if not args.race_globs and not args.eedi_globs:
        print("Nothing to do: provide --race-glob and/or --eedi-glob")

if __name__ == "__main__":
    main()
