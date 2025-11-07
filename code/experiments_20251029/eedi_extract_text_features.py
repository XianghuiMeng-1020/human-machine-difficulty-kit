import argparse, pandas as pd, re, os
from collections import Counter

def text_features(text):
    words = re.findall(r"\w+", text)
    n = len(words)
    uniq = len(set(words))
    avglen = sum(len(w) for w in words)/n if n>0 else 0
    ratio_uniq = uniq / n if n>0 else 0
    long_words = sum(1 for w in words if len(w)>6)
    return {
        "n_words": n,
        "avg_word_len": avglen,
        "uniq_ratio": ratio_uniq,
        "long_word_frac": long_words / n if n>0 else 0,
    }

if __name__=="__main__":
    ap=argparse.ArgumentParser()
    ap.add_argument("--items", default="data/eedi/items.jsonl")
    ap.add_argument("--out", default="analysis/eedi_text_features.csv")
    args=ap.parse_args()

    rows=[]
    with open(args.items, encoding="utf-8") as f:
        for line in f:
            j = eval(line)
            qid = j.get("question_id") or j.get("id")
            txt = (j.get("question_text") or j.get("stem") or "").strip()
            feats = text_features(txt)
            feats["qid"] = qid
            rows.append(feats)

    df = pd.DataFrame(rows)
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    df.to_csv(args.out, index=False)
    print(f"✅ wrote {args.out}, n={len(df)}")
