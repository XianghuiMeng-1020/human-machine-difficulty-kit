import os, hashlib, pandas as pd

IN_CSV = "baselines/bea24_llm_diff/out/all_items_bea_filled.csv"
OUT_CSV = "baselines/bea24_llm_diff/out/all_items_bea_feats.csv"

df = pd.read_csv(IN_CSV)

def fake_llm_feats(text: str):
    h = int(hashlib.md5(text.encode("utf-8")).hexdigest(), 16)
    base = (h % 1000) / 1000.0  # 0~1 稳定数
    return {
        "llm_diff_score": base,
        "llm_len": len(text.split()),
        "llm_is_reasoning": 1 if ("why" in text.lower() or "explain" in text.lower()) else 0,
    }

feat_rows = []
for _, row in df.iterrows():
    text = row.get("question", "") or ""
    feat_rows.append(fake_llm_feats(text))

feat_df = pd.DataFrame(feat_rows)
out_df = pd.concat([df, feat_df], axis=1)

os.makedirs(os.path.dirname(OUT_CSV), exist_ok=True)
out_df.to_csv(OUT_CSV, index=False, encoding="utf-8")
print(f"✅ wrote {OUT_CSV} rows={len(out_df)}")
