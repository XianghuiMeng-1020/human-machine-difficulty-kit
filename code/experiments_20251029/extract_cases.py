import os, sys, csv, json, argparse, glob

def sniff_per_question_files(folder):
    cands = []
    # 1) 优先找含 qid 的 csv/jsonl
    for pat in ["*.csv", "*.jsonl", "*.json"]:
        for p in glob.glob(os.path.join(folder, pat)):
            try:
                if p.endswith(".csv"):
                    with open(p, "r", encoding="utf-8") as f:
                        r = csv.DictReader(f)
                        row = next(r, None)
                        if row and any(k.lower() in ("qid","question_id") for k in row.keys()):
                            cands.append(p)
                elif p.endswith(".jsonl"):
                    with open(p, "r", encoding="utf-8") as f:
                        for line in f:
                            if not line.strip(): continue
                            obj = json.loads(line)
                            if any(k in obj for k in ("qid","question_id")):
                                cands.append(p)
                            break
                elif p.endswith(".json"):
                    obj = json.load(open(p,"r",encoding="utf-8"))
                    if isinstance(obj, list) and obj and isinstance(obj[0], dict):
                        if any(k in obj[0] for k in ("qid","question_id")):
                            cands.append(p)
            except Exception:
                continue
    return list(dict.fromkeys(cands))

def load_rows(path):
    rows=[]
    if path.endswith(".csv"):
        with open(path,"r",encoding="utf-8") as f:
            r=csv.DictReader(f)
            for row in r: rows.append(row)
    elif path.endswith(".jsonl"):
        with open(path,"r",encoding="utf-8") as f:
            for line in f:
                if not line.strip(): continue
                rows.append(json.loads(line))
    elif path.endswith(".json"):
        obj=json.load(open(path,"r",encoding="utf-8"))
        if isinstance(obj,list): rows=obj
    return rows

def get(q, *keys, default=None):
    for k in keys:
        if k in q: return q[k]
        # 尝试大小写/中文键
        for kk in q.keys():
            if kk.lower()==k.lower(): return q[kk]
    return default

def coerce_float(x, default=None):
    try:
        return float(x)
    except:
        return default

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--folder", default="data/eedi_gpt4o_300x1")
    ap.add_argument("--tau", type=float, default=0.8)
    ap.add_argument("--out_prefix", default="analysis/eedi_gpt4o_tau08")
    args=ap.parse_args()

    os.makedirs(os.path.dirname(args.out_prefix), exist_ok=True)

    files = sniff_per_question_files(args.folder)
    if not files:
        print("No per-question files found in", args.folder)
        sys.exit(1)

    # 选一个最可能的（优先 scores.csv）
    files.sort(key=lambda p: (0 if os.path.basename(p).startswith("scores") else 1, len(p)))
    picked = files[0]
    print("Using per-question file:", picked)

    rows = load_rows(picked)

    human_easy = {"简单H","H_easy","easy","Easy","简单"}
    human_mid  = {"中等H","H_mid","mid","Middle","中等"}
    human_hard = {"困难H","H_hard","hard","Hard","困难"}

    out_h_easy_m_hard=[]
    out_h_hard_m_easy=[]

    for q in rows:
        qid = str(get(q,"qid","question_id", default=""))
        if not qid: continue

        # 人类标签尝试字段
        h = get(q,"human_bucket","H_label","human_label","human_difficulty","difficulty", default="")
        h = str(h)

        # 模型标签（如果已有某阈值标注列）
        m = get(q,"M_label","model_bucket","model_label","M_tau","model_tau", default="")

        # 缺少模型标签就用 prob+correct 打标
        if not m:
            prob = coerce_float(get(q,"prob_max","prob","confidence","conf","prob_chosen"), default=None)
            corr = get(q,"correct","is_correct","label_correct","acc")
            try:
                corr = int(corr)
            except:
                corr = 1 if str(corr).lower() in ("1","true","yes") else 0
            if prob is None:
                # 尝试用 logits/probs 列
                prob = coerce_float(get(q,"p_max","p"), default=0.0)
            if prob >= args.tau and corr==1:
                m="简单M"
            elif prob >= args.tau and corr==0:
                m="困难M"
            else:
                m="中等M"

        # 标准化人类标签
        if h in human_easy:
            h_std="简单H"
        elif h in human_mid:
            h_std="中等H"
        elif h in human_hard:
            h_std="困难H"
        else:
            h_std="MissingH"

        # 两类清单
        if h_std=="简单H" and m=="困难M":
            out_h_easy_m_hard.append(qid)
        if h_std=="困难H" and m=="简单M":
            out_h_hard_m_easy.append(qid)

    # 写出
    je = args.out_prefix + "_human_easy_model_hard.json"
    jh = args.out_prefix + "_human_hard_model_easy.json"
    json.dump(out_h_easy_m_hard, open(je,"w",encoding="utf-8"), ensure_ascii=False, indent=2)
    json.dump(out_h_hard_m_easy, open(jh,"w",encoding="utf-8"), ensure_ascii=False, indent=2)
    print(f"✓ 人易机难: {len(out_h_easy_m_hard)}  → {je}")
    print(f"✓ 人难机易: {len(out_h_hard_m_easy)}  → {jh}")

if __name__=="__main__":
    main()
