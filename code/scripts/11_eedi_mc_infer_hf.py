
import os, time, math, json, argparse, torch, pandas as pd, numpy as np
from pathlib import Path
from transformers import AutoTokenizer, AutoModelForMultipleChoice

REQ_COLS = ["qid","question","choice_A","choice_B","choice_C","choice_D"]
OPT_COLS = ["context","gold_choice"]
CHOICE_KEYS = ["A","B","C","D"]

def _load_items(path):
    df = pd.read_csv(path, dtype=str)
    miss = [c for c in REQ_COLS if c not in df.columns]
    if miss:
        raise SystemExit(f"[ERR] {path} 缺列: {miss}")
    for c in REQ_COLS+OPT_COLS:
        if c in df.columns: df[c] = df[c].fillna("")
    return df

def _encode_mc(tok, batch, max_len=512):
    contexts  = batch["context"].tolist() if "context" in batch.columns else [""]*len(batch)
    questions = batch["question"].tolist()
    choices   = [batch[f"choice_{k}"].tolist() for k in CHOICE_KEYS]
    texts=[]
    for i in range(len(questions)):
        cxt = contexts[i].strip(); q = questions[i].strip()
        prompt = (cxt + " " + q).strip() if cxt else q
        for k_idx in range(4):
            texts.append((prompt, choices[k_idx][i]))
    enc = tok([t[0] for t in texts], [t[1] for t in texts],
              padding=True, truncation=True, max_length=max_len, return_tensors="pt")
    bsz = len(questions)
    for k in enc:
        enc[k] = enc[k].view(bsz, 4, -1)
    return enc

def _logits_to_probs(logits):
    z = logits - logits.max(dim=1, keepdim=True).values
    e = torch.exp(z)
    return e / e.sum(dim=1, keepdim=True)

def run(model_name, items_csv, out_csv, batch_size=8, device=None):
    t0=time.time()
    Path(out_csv).parent.mkdir(parents=True, exist_ok=True)
    df=_load_items(items_csv)

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    tok = AutoTokenizer.from_pretrained(model_name, use_fast=True)
    mdl = AutoModelForMultipleChoice.from_pretrained(model_name).to(device)
    mdl.eval()

    rows=[]
    for i in range(0, len(df), batch_size):
        chunk = df.iloc[i:i+batch_size].copy()
        enc = _encode_mc(tok, chunk)
        enc = {k:v.to(device) for k,v in enc.items()}
        with torch.no_grad():
            out = mdl(**enc)
            logits = out.logits
            probs  = _logits_to_probs(logits).cpu().numpy()

        for j in range(len(chunk)):
            qid = str(chunk.iloc[j]["qid"])
            prob_map  = {f"prob_{k}": float(probs[j, idx]) for idx,k in enumerate(CHOICE_KEYS)}
            logit_map = {f"logit_{k}": float(logits[j, idx].detach().cpu().item()) for idx,k in enumerate(CHOICE_KEYS)}
            pred_idx = int(np.argmax(probs[j])); pred_choice = CHOICE_KEYS[pred_idx]
            gold = str(chunk.iloc[j]["gold_choice"]).strip().upper() if "gold_choice" in chunk.columns else ""
            is_corr = (pred_choice==gold) if gold in CHOICE_KEYS else None
            conf = float(probs[j, pred_idx])
            rows.append({
                "qid": qid, **prob_map, **logit_map,
                "pred_choice": pred_choice, "gold_choice": gold,
                "is_correct": (1.0 if is_corr is True else (0.0 if is_corr is False else "")),
                "conf": conf
            })
        if (i//batch_size) % 10 == 0:
            print(f"[{min(i+batch_size,len(df))}/{len(df)}] ...")

    out = pd.DataFrame(rows)
    front = ["qid","pred_choice","gold_choice","is_correct","conf"]
    out = out[front + [c for c in out.columns if c not in front]]
    out.to_csv(out_csv, index=False)
    print(f"[WROTE] {out_csv} rows={len(out)} uniq_conf={out['conf'].dropna().nunique()} time={time.time()-t0:.1f}s")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--alias", required=True)
    ap.add_argument("--items", default="data/eedi_raw_items.csv")
    ap.add_argument("--batch_size", type=int, default=8)
    args = ap.parse_args()
    out_csv = f"experiments_20251029/results/eedi/{args.alias}/round1/per_question.csv"
    run(args.model, args.items, out_csv, batch_size=args.batch_size)
