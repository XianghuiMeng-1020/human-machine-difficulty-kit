#!/usr/bin/env python3
import argparse, json, os, re
import pandas as pd

PROC_PATH = "data/race/processed/race_test.jsonl"

def infer_from_prefix(qid: str | None) -> str | None:
    if not isinstance(qid, str):
        return None
    head = qid.split(":", 1)[0].lower()
    return head if head in ("high", "middle") else None

def load_processed_maps():
    """Build multiple lookup maps from the processed jsonl."""
    maps = {
        "full": {},        # exact ids -> difficulty
        "pair": {},        # "level:article" -> difficulty
        "article": {},     # article id only -> difficulty
    }
    if not os.path.exists(PROC_PATH):
        return maps

    with open(PROC_PATH, encoding="utf-8") as f:
        for line in f:
            try:
                ex = json.loads(line)
            except Exception:
                continue
            diff = ex.get("difficulty")
            if diff not in ("high", "middle"):
                continue

            cand = []
            # recorded keys
            for k in ("question_id", "id"):
                v = ex.get(k)
                if isinstance(v, str) and v:
                    cand.append(v)

            # derive variants from question_id if present:
            qid = ex.get("question_id")
            if isinstance(qid, str):
                parts = qid.split(":")
                if len(parts) >= 2:
                    maps["pair"][":".join(parts[:2])] = diff   # level:article
                    maps["article"][parts[1]] = diff          # article only

            # also from id if it contains colon-separated form
            rid = ex.get("id")
            if isinstance(rid, str):
                parts = rid.split(":")
                if len(parts) >= 2:
                    maps["pair"][":".join(parts[:2])] = diff
                    maps["article"][parts[1]] = diff

            for c in cand:
                maps["full"][c] = diff

    return maps

def best_guess(qid: object, maps) -> str | None:
    """Try several variants to recover difficulty for a given question_id string."""
    if not isinstance(qid, str):
        return None

    # 1) prefix
    d = infer_from_prefix(qid)
    if d: return d

    # 2) exact full match (scores.qid equals processed.question_id/id)
    if qid in maps["full"]:
        return maps["full"][qid]

    # 3) strip trailing ':<digits>' (question index), try full again
    base = re.sub(r":[0-9]+$", "", qid)
    if base in maps["full"]:
        return maps["full"][base]

    # 4) try pair 'level:article' if qid looks like 'level:article[:idx]'
    parts = qid.split(":")
    if len(parts) >= 2:
        pair = ":".join(parts[:2])
        if pair in maps["pair"]:
            return maps["pair"][pair]
        # also try 'article' only
        if parts[1] in maps["article"]:
            return maps["article"][parts[1]]

    # 5) last resort: if qid is just article id and we have article map
    if qid in maps["article"]:
        return maps["article"][qid]

    return None

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scores", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    df = pd.read_csv(args.scores)
    # if difficulty already there, keep it; otherwise create empty
    if "difficulty" not in df.columns:
        df["difficulty"] = pd.NA

    maps = load_processed_maps()
    miss = df["difficulty"].isna()
    if miss.any():
        df.loc[miss, "difficulty"] = df.loc[miss, "question_id"].apply(lambda q: best_guess(q, maps))

    df.to_csv(args.out, index=False)
    print(f"Wrote -> {args.out}")
    try:
        print(df["difficulty"].value_counts(dropna=False))
    except Exception:
        pass

if __name__ == "__main__":
    main()
