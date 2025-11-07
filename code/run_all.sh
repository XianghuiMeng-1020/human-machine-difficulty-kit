#!/usr/bin/env bash
set -euo pipefail

# -------- config --------
PY=${PY:-python}
MODELS=${MODELS:-"stage3 robertaL drobertaB"}   # 需要推理的模型名（与你脚本中读取的一致）
DEVICE=${DEVICE:-cpu}                           # cpu / cuda
BINS=${BINS:-15}                                # 可靠性分箱
# ------------------------

t() { date "+%Y-%m-%d %H:%M:%S"; }
log() { printf "[%s] %s\n" "$(t)" "$*"; }

log "Step 0 — 环境检查"
$PY - <<'PY'
import importlib, sys
need = ["pandas","numpy","matplotlib","scipy","statsmodels","scikit-learn","transformers","torch"]
miss=[m for m in need if importlib.util.find_spec(m) is None]
assert not miss, f"缺少依赖: {miss}"
print("[OK] 依赖检查通过。Python", sys.version.split()[0])
PY

log "Step 1 —（可选）多模型推理/补conf（如已有 per_question.csv 会自动跳过）"
$PY - <<PY
import os, sys
MODELS = "${MODELS}".split()
os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY","1")
# 运行你现有的多模型推理脚本（会自动跳过已有 conf 的）
os.system(f"{sys.executable} - <<'PY2'\n" + r'''
import os, pandas as pd, glob, time, numpy as np
from pathlib import Path
from transformers import AutoTokenizer, AutoModelForMultipleChoice
import torch

def has_conf_csv(p):
    try: df=pd.read_csv(p, nrows=3)
    except: return False
    return "conf" in df.columns and pd.to_numeric(df["conf"], errors="coerce").notna().any()

def ensure_dir(p): Path(p).parent.mkdir(parents=True, exist_ok=True)

raw_csv="data/race_raw_items.csv"
if not os.path.exists(raw_csv):
    print("[INFO] 下载 RACE 原始数据…")
    # 你已有的获取逻辑，这里假设已存在；不再重复实现
    pass

items=pd.read_csv(raw_csv)
def run_one(model_name, tag):
    out_csv=f"experiments_20251029/results/race/{tag}/round1/per_question.csv"
    if os.path.exists(out_csv) and has_conf_csv(out_csv):
        print(f"[SKIP] {tag} 已有 per_question.csv 且含 conf，跳过推理。")
        return
    print(f"[LOAD] {tag} -> ${DEVICE}")
    tok=AutoTokenizer.from_pretrained(model_name, use_fast=True)
    mdl=AutoModelForMultipleChoice.from_pretrained(model_name)
    mdl.to("${DEVICE}")
    rows=[]
    t0=time.time()
    with torch.no_grad():
        for i, row in items.iterrows():
            context=str(row["context"]); question=str(row["question"])
            choices=[str(row[f"choice_{c}"]) for c in "ABCD"]
            enc=tok([context]*4, [question+" "+c for c in choices],
                    truncation=True, padding=True, return_tensors="pt")
            enc={k:v.to("${DEVICE}") for k,v in enc.items()}
            logits=mdl(**enc).logits.squeeze(0).detach().cpu().numpy()
            e=np.exp(logits-logits.max()); prob=e/e.sum()
            pred="ABCD"[int(prob.argmax())]
            conf=float(prob.max())
            rows.append({
                "qid":row["qid"],
                "gold_choice":row.get("gold_choice",""),
                "pred_choice":pred,
                "is_correct": float(pred==str(row.get("gold_choice",""))),
                "conf": conf,
                "prob_A": float(prob[0]), "prob_B": float(prob[1]),
                "prob_C": float(prob[2]), "prob_D": float(prob[3]),
            })
            if (i+1)%50==0: print(f"[{tag}] {i+1}/{len(items)}")
    out=pd.DataFrame(rows)
    ensure_dir(out_csv); out.to_csv(out_csv, index=False)
    print(f"[WROTE] {out_csv} rows={len(out)} uniq_conf={out['conf'].nunique(dropna=True)} time={time.time()-t0:.1f}s")

models_map={"stage3":"microsoft/deberta-v3-large",
            "robertaL":"roberta-large",
            "drobertaB":"distilroberta-base"}
for tag in "${MODELS}".split():
    run_one(models_map.get(tag, tag), tag)
''' + "\nPY2")
PY

log "Step 2 — 聚合 00/01/02/03 + 02b"
$PY scripts/00_build_from_filelist.py
$PY scripts/01_continuous_alignment_and_logit.py
$PY scripts/02_misalignment_and_tau.py
$PY scripts/02b_misalignment_significance.py
$PY scripts/03_calibration_auc_and_ci.py

log "Step 3 — 快速总览"
$PY - <<'PY'
import pandas as pd, glob, os
def peek(p, n=8):
    if os.path.exists(p):
        print(f"\n== {p} =="); print(pd.read_csv(p).head(n).to_string(index=False))
for p in [
    "tables/race_model_comparison_summary.csv",
    "tables/race_misalignment_with_ci.csv",
    "tables/race_conf_discrimination_auc.csv",
    "tables/race_calibration_summary.csv",
    "tables/race_continuous_alignment_correlations_CI.csv",
]:
    peek(p, 10)
print("\n[FIGS]", [os.path.basename(x) for x in sorted(glob.glob('figures/race_*.*'))])
PY

log "All done."
