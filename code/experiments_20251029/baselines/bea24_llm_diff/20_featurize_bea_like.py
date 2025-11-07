import os, json, pandas as pd
from collections import Counter

SRC = "baselines/bea24_llm_diff/out/all_items_bea_like.csv"
OUTDIR = "baselines/bea24_llm_diff/out"
os.makedirs(OUTDIR, exist_ok=True)

df = pd.read_csv(SRC)

feat_rows = []
for r in df.itertuples(index=False):
    stem = (r.stem or "")
    toks = stem.split()
    n_tok = len(toks)
    n_char = len(stem)
    n_sent = stem.count(".") + stem.count("。") + stem.count("!") + stem.count("?")

    try:
        opts = json.loads(r.options)
    except Exception:
        opts = []
    n_opts = len(opts)
    opt_lens = [len(o.split()) for o in opts] if opts else []
    opt_len_var = float(pd.Series(opt_lens).std()) if opt_lens else 0.0

    feat_rows.append({
        "item_id": r.item_id,
        "dataset": r.dataset,
        "n_tok": n_tok,
        "n_char": n_char,
        "n_sent": n_sent,
        "n_opts": n_opts,
        "opt_len_var": opt_len_var,
        "target_diff": r.target_diff,
    })

feat_df = pd.DataFrame(feat_rows)
feat_df.to_csv(os.path.join(OUTDIR, "all_items_bea_feats.csv"), index=False)
print("✅ wrote", os.path.join(OUTDIR, "all_items_bea_feats.csv"), "rows=", len(feat_df))
