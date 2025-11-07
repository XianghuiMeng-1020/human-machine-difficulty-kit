import os, re, json
from pathlib import Path
import pandas as pd
import numpy as np

ROOT  = Path("experiments_20251029/results/eedi")
DATA  = Path("data/eedi")
META  = DATA/"metadata/question_metadata_task_3_4.csv"
TRAIN = DATA/"train_data/train_task_3_4.csv"

CANDS = [
    DATA/"processed/task34_for_llm.full.final.jsonl",
    DATA/"processed/task34_for_llm.full.jsonl",
    DATA/"processed/task34_for_llm.final.jsonl",
    DATA/"processed/task34_for_llm.jsonl",
    DATA/"processed/task34_for_llm.mini10.jsonl",
]

CHOICES = ["A","B","C","D"]

def norm_choice(x):
    if x is None: return None
    s=str(x).strip().upper()
    if s in CHOICES: return s
    # 尝试抽取自由文本中的 A/B/C/D
    m=re.search(r'\b([ABCD])\b', s)
    if m: return m.group(1)
    # 兼容 "(A)"/"Answer: C"/"选项 B"
    m=re.search(r'\(?\s*([ABCD])\s*\)?', s)
    if m: return m.group(1)
    # 索引数字
    if s.isdigit():
        i=int(s)
        if 0<=i<4: return CHOICES[i]
    return None

def soften(v):
    if not isinstance(v,(list,tuple)) or len(v)==0: return None
    arr=np.array(v,dtype=float)
    # 判定像 logits 就 softmax；像概率就归一
    if np.any(arr<0) or np.any(arr>1) or arr.sum()==0 or arr.sum()>1.5:
        arr=np.exp(arr - arr.max())
        s=arr.sum(); arr=arr/(s if s>0 else 1.0)
    else:
        s=arr.sum(); arr=arr/(s if s>0 else 1.0)
    return arr

def load_answer_key():
    # 优先 metadata
    if META.exists():
        meta=pd.read_csv(META)
        q = next((c for c in meta.columns if c.lower() in ["qid","question_id","question"]), None)
        g = next((c for c in meta.columns if "correct" in c.lower() and any(k in c.lower() for k in ["option","choice","answer"])), None)
        if q and g:
            ans=meta[[q,g]].copy(); ans.columns=["qid","gold_choice"]
            ans["qid"]=ans["qid"].astype(str)
            ans["gold_choice"]=ans["gold_choice"].map(norm_choice)
            ans=ans.dropna(subset=["gold_choice"])
            if len(ans): 
                print(f"[answer from META] rows={len(ans)}"); 
                return ans
    # 兜底 train
    if TRAIN.exists():
        tr=pd.read_csv(TRAIN)
        q = next((c for c in tr.columns if c.lower() in ["qid","question_id","question"]), None)
        g = next((c for c in tr.columns if "correct" in c.lower() and any(k in c.lower() for k in ["option","choice","answer"])), None)
        if q and g:
            ans=tr[[q,g]].drop_duplicates().copy(); ans.columns=["qid","gold_choice"]
            ans["qid"]=ans["qid"].astype(str)
            ans["gold_choice"]=ans["gold_choice"].map(norm_choice)
            ans=ans.dropna(subset=["gold_choice"])
            if len(ans): 
                print(f"[answer from TRAIN] rows={len(ans)}"); 
                return ans
    raise RuntimeError("没有可用的答案键（需要包含 qid 与 正确选项 列）")

def model_name_from_path(p:Path):
    s=p.name.lower()
    if "full.final" in s: return "full.final"
    if s.endswith(".full.jsonl"): return "full"
    if "final.jsonl" in s and "full" not in s: return "final"
    if "mini10" in s: return "mini10"
    return "jsonl"

def get_any(d, keys):
    for k in keys:
        if isinstance(k, tuple): # 嵌套路径
            cur=d
            ok=True
            for kk in k:
                if isinstance(cur, dict) and kk in cur:
                    cur=cur[kk]
                else:
                    ok=False; break
            if ok: return cur
        else:
            if isinstance(d, dict) and k in d: return d[k]
    return None

def extract_pred_and_conf(r):
    # 可能的“预测”路径 & 文本容器
    pred_cands = [
        "pred","prediction","answer","label","choice","final_answer","final","output",
        ("output","choice"),("output","label"),("output","answer"),("output","final_answer"),("output","text"),
        ("result","choice"),("result","label"),("result","answer"),
        ("response","choice"),("response","label"),("response","answer"),
        ("model_answer",),("text",),
    ]
    raw = get_any(r, pred_cands)
    # 如果是 dict，尝试嵌套 choice/label
    if isinstance(raw, dict):
        raw = get_any(raw, ["choice","label","answer","final_answer","text"])
    # 如果是 list 且每项是 dict，看看第一个的 choice/label
    if isinstance(raw, list) and raw and isinstance(raw[0], dict):
        raw = get_any(raw[0], ["choice","label","answer","final_answer","text"])
    pred_choice = norm_choice(raw)

    # 可能的“概率/置信”路径
    conf = None
    # 直接数值
    num_keys = ["conf","confidence","prob","proba","score"]
    v = get_any(r, num_keys)
    if isinstance(v,(int,float)): conf=float(v)
    # 列表形式
    if conf is None:
        arr = get_any(r, ["probs","probabilities","logits","logprobs","option_scores","scores","pred_logits","choice_logits"])
        arr = soften(arr)
        if isinstance(arr, np.ndarray):
            if pred_choice in ["A","B","C","D"]:
                idx = {"A":0,"B":1,"C":2,"D":3}[pred_choice]
                if 0<=idx<len(arr):
                    conf = float(arr[idx])
                else:
                    conf = float(arr.max())
            else:
                conf = float(arr.max())
    # 若还没有，尝试从 A/B/C/D 概率字段推断
    if conf is None:
        # 例如 {"A":0.1,"B":0.6,...}
        abcd = {k:r.get(k) for k in CHOICES if isinstance(r.get(k),(int,float))}
        if abcd:
            if pred_choice in abcd and isinstance(abcd[pred_choice], (int,float)):
                conf=float(abcd[pred_choice])
            else:
                conf=float(max(abcd.values()))
                # 顺便补 pred_choice
                if pred_choice is None:
                    pred_choice = max(abcd, key=abcd.get)

    # 最后的兜底：如果还是没有 pred_choice，尝试在自由文本里抓
    if pred_choice is None:
        txt = get_any(r, ["text","response","model_answer","final_answer","output","answer"])
        if isinstance(txt, dict):
            txt = get_any(txt, ["text","response","model_answer","final_answer","choice","label"])
        if isinstance(txt, list):
            txt = " ".join(map(str, txt[:3]))
        if isinstance(txt, str):
            m=re.search(r'(?i)(answer|final|预测|选项)\s*[:：]?\s*\(?\s*([ABCD])\s*\)?', txt)
            if not m: m=re.search(r'\b([ABCD])\b', txt)
            if m: pred_choice = m.group(2) if m.lastindex and m.lastindex>=2 else m.group(1)

    return pred_choice, conf

def parse_jsonl_to_perq(path:Path, ans_key:pd.DataFrame):
    rec=[]
    with open(path,'r',encoding='utf-8') as f:
        for line in f:
            line=line.strip()
            if not line: continue
            try:
                r=json.loads(line)
            except: 
                continue
            qid = r.get("qid") or r.get("question_id") or r.get("id") or r.get("question")
            if qid is None: 
                continue
            qid=str(qid)
            pred_choice, conf = extract_pred_and_conf(r)
            rec.append((qid, pred_choice, conf))
    if not rec:
        return None
    df=pd.DataFrame(rec, columns=["qid","pred_choice","conf"])
    df["qid"]=df["qid"].astype(str)
    out=df.merge(ans_key, on="qid", how="left")
    out["gold_choice"]=out["gold_choice"].apply(norm_choice)
    out["is_correct"]=(out["pred_choice"]==out["gold_choice"]).astype(int)
    return out[["qid","pred_choice","gold_choice","is_correct","conf"]]

def main():
    ans_key=load_answer_key()
    wrote=False
    for p in CANDS:
        p=Path(p)
        if not p.exists(): 
            print("[MISS]", p); 
            continue
        out=parse_jsonl_to_perq(p, ans_key)
        if out is None or out.empty:
            print("NO PRED ROWS in", p); 
            continue
        model=model_name_from_path(p)
        outdir=ROOT/model/"round1"
        outdir.mkdir(parents=True, exist_ok=True)
        out.to_csv(outdir/"per_question.csv", index=False)
        acc=out["is_correct"].mean() if len(out) else float('nan')
        print(f"WROTE {outdir/'per_question.csv'} rows={len(out)} acc={acc:.3f} model={model}")
        print(out.head(3).to_string(index=False))
        wrote=True
    if not wrote:
        print("没有生成任何 per_question.csv —— 检查 JSONL 是否含有 qid 与预测。")

if __name__=="__main__":
    main()
