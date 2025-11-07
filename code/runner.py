from __future__ import annotations

import os
import re
import argparse
import hashlib
import random
import numpy as np
import pandas as pd
from tqdm import tqdm

from src.utils.io import read_jsonl, write_jsonl  # noqa: F401 (write_jsonl reserved for future use)
from src.scoring.dummy_client import DummyClient
from src.scoring.openai_client_stub import OpenAIClient
from src.scoring.prompts import build_prompt
from src.analysis.datamap import summarize_scores, assign_regions
from src.analysis.calibration import ece_bin, brier, temperature_scale  # noqa: F401 (direct CLI call uses)
from src.analysis.alignment import human_machine_crosstab, rank_corr  # noqa: F401 (direct CLI call uses)

LETTER_SET = ("A", "B", "C", "D")


def build_old2new_map(options_orig: dict, options_perm: dict) -> dict:
    """
    Map original letter -> displayed letter after permutation via text matching.
    This is robust to how options_perm was built (by content instead of letter).
    """
    txt2new = {txt: L for L, txt in options_perm.items()}
    old2new = {}
    for old in LETTER_SET:
        txt = options_orig.get(old)
        if txt in txt2new:
            old2new[old] = txt2new[txt]
    return old2new

def _extract_gold_letter(ex: dict, options_orig: dict) -> str | None:
    """
    Robustly extract the correct option as A/B/C/D from various schemas.
    - Supports letter fields (A-D)
    - Supports digit fields (1-4 and 0-3)
    - Falls back to matching correct answer text against options
    """
    def map_digit(v: str) -> str | None:
        m = re.fullmatch(r"\s*([1-4])\s*", v)
        if m:  # 1-4 -> A-D
            return "ABCD"[int(m.group(1)) - 1]
        m = re.fullmatch(r"\s*([0-3])\s*", v)
        if m:  # 0-3 -> A-D
            return "ABCD"[int(m.group(1))]
        return None

    # 1) common letter-like fields
    for k in [
        "answer", "correct_letter", "CorrectLetter", "correctLetter",
        "CorrectAnswer", "correct_answer", "AnswerValue", "gold", "gold_letter",
        "label", "target", "solution", "gt"
    ]:
        if k in ex and ex[k] is not None:
            s = str(ex[k]).strip()
            L = _norm_letter(s)
            if L in LETTER_SET:
                return L
            L = map_digit(s)
            if L in LETTER_SET:
                return L

    # 2) index-like fields (1-4 or 0-3)
    for k in [
        "answer_idx", "answer_index", "correct_idx", "correct_index",
        "CorrectIndex", "index", "gold_idx", "gold_index"
    ]:
        if k in ex and ex[k] is not None:
            s = str(ex[k]).strip()
            L = map_digit(s)
            if L in LETTER_SET:
                return L

    # 3) match by text if a correct answer text is present
    for k in ["correct_text", "correct_answer_text", "CorrectAnswerText", "gold_text"]:
        if k in ex and ex[k]:
            txt = str(ex[k]).strip()
            if txt:
                for L, opt in options_orig.items():
                    if str(opt).strip() == txt:
                        return L

    return None

def _norm_letter(x: object) -> str | None:
    """Normalize an answer hint into A/B/C/D if possible."""
    if x is None:
        return None
    s = str(x).strip().upper()

    # 1) Direct letter first (A/B/C/D anywhere)
    m = re.search(r"\b([ABCD])\b", s)
    if m:
        return m.group(1)

    # 2) Common patterns like "Answer: C" / "Option B"
    m = re.search(r"(ANSWER|OPTION)[^A-Z0-9]*([ABCD])\b", s)
    if m:
        return m.group(2)

    # 3) Map digits to letters (1-4 => A-D)
    m = re.search(r"\b([1-4])\b", s)
    if m:
        return "ABCD"[int(m.group(1)) - 1]

    # 4) Also tolerate zero-based (0-3 => A-D)
    m = re.search(r"\b([0-3])\b", s)
    if m:
        return "ABCD"[int(m.group(1))]

    return None


def _load_questions(dataset: str, split: str):
    if dataset == "race":
        path = f"data/race/processed/race_{split}.jsonl"
        if not os.path.exists(path):
            raise FileNotFoundError(
                f"Missing {path}. Please run: python -m src.data.prepare_race --split {split}"
            )
        return list(read_jsonl(path))
    elif dataset == "eedi":
        path = "data/eedi/processed/task34_for_llm.jsonl"
        if not os.path.exists(path):
            raise FileNotFoundError(
                "Missing data/eedi/processed/task34_for_llm.jsonl. "
                "Please run: python -m src.data.prepare_eedi "
                "--in data/eedi/raw/interactions_task34.csv --out data/eedi/processed"
            )
        return list(read_jsonl(path))
    else:
        raise ValueError("dataset must be one of: race | eedi")


def _make_client(model_name: str):
    if model_name == "dummy":
        return DummyClient()
    elif model_name.startswith("openai"):
        model = model_name.split(":", 1)[1] if ":" in model_name else "gpt-4o"
        return OpenAIClient(model=model)
    else:
        raise NotImplementedError(f"unknown model: {model_name}")


def _permute_options(options: dict[str, str], seed: int):
    """
    Deterministically shuffle present option letters (subset of A-D) with a seed.
    Returns (options_perm, map_orig_to_perm, map_perm_to_orig).
    """
    letters = [k for k in ["A", "B", "C", "D"] if k in options]
    rng = random.Random(seed)
    perm = letters[:]
    rng.shuffle(perm)
    map_o2p = {letters[i]: perm[i] for i in range(len(letters))}
    map_p2o = {v: k for k, v in map_o2p.items()}
    options_perm = {map_o2p[L]: options[L] for L in letters}
    return options_perm, map_o2p, map_p2o


def cmd_demo(args):
    os.makedirs("outputs/demo", exist_ok=True)
    import json

    qpath = "sample_data/race_demo.jsonl"
    with open(qpath, encoding="utf-8") as f:
        qs = [json.loads(l) for l in f]

    client = DummyClient(seed=42)
    recs = []
    for r in range(args.rounds):
        for ex in qs:
            prompt = build_prompt(ex.get("passage", ""), ex["question"], ex["options"])
            sr = client.score_mcq(prompt, ex["options"])
            p_correct = sr.probs.get(ex["answer"], 0.0)
            recs.append(
                {
                    "question_id": ex["id"],
                    "run_id": r,
                    "difficulty": ex.get("difficulty"),
                    "chosen": sr.chosen,
                    "p_correct": p_correct,
                    "correct": int(sr.chosen == ex["answer"]),
                    "correct_letter": gold,
                }
            )
    df = pd.DataFrame(recs)
    df.to_csv("outputs/demo/scores.csv", index=False)

    summ = summarize_scores(df)
    summ.to_csv("outputs/demo/datamap.csv", index=False)
    print("Demo complete. See outputs/demo/")


def cmd_score(args):
    os.makedirs(args.out, exist_ok=True)
    qs = _load_questions(args.dataset, args.split)
    client = _make_client(args.model)

    recs = []
    for r in range(args.rounds):
        for ex in tqdm(qs, desc=f"round {r}"):
            passage = ex.get("passage", None)
            question = ex.get("question") or ex.get("stem") or ""

            # options may appear as dict or A/B/C/D fields; fall back to letters if missing
            options = ex.get("options")
            if options is None:
                cand = {L: ex.get(L) for L in LETTER_SET if ex.get(L) is not None}
                options = cand if cand else {"A": "A", "B": "B", "C": "C", "D": "D"}

            qid = ex.get("id") or ex.get("question_id") or ""
            gold = _extract_gold_letter(ex, options)

            # keep a copy of original options for mapping/debug
            options_orig = dict(options)

            # permutation control (PERMUTE=0 keeps original A-D order)
            seed_src = f"{qid}-{r}"
            seed = int(hashlib.sha256(seed_src.encode()).hexdigest(), 16) % (2**32)
            do_perm = os.getenv("PERMUTE", "1") != "0"
            if do_perm:
                options_perm, o2p, p2o = _permute_options(options_orig, seed)
            else:
                options_perm = {L: options_orig[L] for L in ["A", "B", "C", "D"] if L in options_orig}
                o2p = {L: L for L in options_perm.keys()}
                p2o = o2p.copy()

            prompt = build_prompt(passage, question, options_perm)

            # image support is handled inside the client via USE_IMAGE env
            try:
                sr = client.score_mcq(prompt, options_perm, image_path=ex.get("image_path"))
            except TypeError:
                sr = client.score_mcq(prompt, options_perm)

            # chosen in perm space -> original space
            choice_perm = _norm_letter(sr.chosen)
            choice_orig = p2o.get(choice_perm, choice_perm) if choice_perm else None

            # probs in perm space -> original space
            raw_probs = sr.probs or {}
            probs_orig = {"A": np.nan, "B": np.nan, "C": np.nan, "D": np.nan}
            for L in LETTER_SET:
                Lp = o2p.get(L)
                if Lp is not None and Lp in raw_probs:
                    try:
                        probs_orig[L] = float(raw_probs[Lp])
                    except Exception:
                        probs_orig[L] = np.nan

            p_correct = (probs_orig.get(gold) if gold else np.nan)
            p_chosen = (probs_orig.get(choice_orig) if choice_orig else np.nan)

            recs.append(
                {
                    "question_id": qid,
                    "run_id": r,
                    # gold / predicted in ORIGINAL letter space
                    "correct_letter": gold,
                    "chosen": choice_orig,
                    # probabilities in ORIGINAL space
                    "p_correct": p_correct,
                    "p_chosen": p_chosen,
                    "correct": (int(choice_orig == gold) if (gold and choice_orig) else np.nan),
                    "p_A": probs_orig.get("A", np.nan),
                    "p_B": probs_orig.get("B", np.nan),
                    "p_C": probs_orig.get("C", np.nan),
                    "p_D": probs_orig.get("D", np.nan),
                }
            )

    df = pd.DataFrame(recs)
    path = os.path.join(args.out, "scores.csv")
    df.to_csv(path, index=False)
    print(f"Wrote -> {path}")

def cmd_datamap(args):
    df = pd.read_csv(args.inp)
    df = df.dropna(subset=["p_correct", "correct"])
    summ = summarize_scores(df)
    summ = assign_regions(summ, method="quantile" if args.quantile else "fixed")
    os.makedirs(args.out, exist_ok=True)
    summ.to_csv(os.path.join(args.out, "datamap.csv"), index=False)
    print("Data Map saved.")


def cmd_calibrate(args):
    import json

    df = pd.read_csv(args.inp).dropna(subset=["p_correct", "correct"])

    df = df.sample(frac=1.0, random_state=42).reset_index(drop=True)
    n = len(df)
    df_tr = df.iloc[: int(0.8 * n)]
    df_te = df.iloc[int(0.8 * n) :]

    y_tr = df_tr["correct"].values.astype(int)
    p_tr = df_tr["p_correct"].values
    y_te = df_te["correct"].values.astype(int)
    p_te = df_te["p_correct"].values

    ts = temperature_scale(p_tr, y_tr)

    eps = 1e-12
    logits_te = np.log(np.clip(p_te, eps, 1 - eps)) - np.log(np.clip(1 - p_te, eps, 1 - eps))
    p_te_cal = 1 / (1 + np.exp(-logits_te / ts.T))

    metrics = {
        "ece_before": ece_bin(y_te, p_te, 10),
        "brier_before": brier(y_te, p_te),
        "ece_after": ece_bin(y_te, p_te_cal, 10),
        "brier_after": brier(y_te, p_te_cal),
        "T": ts.T,
        "nll_before": ts.nll_before,
        "nll_after": ts.nll_after,
        "n_test": int(len(y_te)),
    }

    os.makedirs(args.out, exist_ok=True)
    with open(os.path.join(args.out, "calibration.json"), "w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)
    print("Calibration saved.")


def cmd_align(args):
    import os
    import pandas as pd
    from src.analysis.alignment import rank_corr

    scores = pd.read_csv(args.scores)
    if not {"question_id", "p_correct", "correct"}.issubset(scores.columns):
        raise RuntimeError("scores.csv must have columns: question_id, p_correct, correct")

    agg = (
        scores.groupby("question_id", as_index=False)
        .agg(mean_p=("p_correct", "mean"), acc=("correct", "mean"))
    )

    human = pd.read_csv(args.human)

    candidates = ["diff_conf", "wrong_conf", "diff_conf_old", "error_rate"]
    chosen = None
    for c in candidates:
        if c in human.columns and human[c].notna().sum() > 0:
            chosen = c
            break

    if chosen is None:
        raise RuntimeError(
            f"No usable human difficulty column found in {args.human}. "
            f"Expected one of {candidates} with at least one non-NaN value."
        )

    human_sel = human[["question_id", chosen]].rename(columns={chosen: "human_difficulty"})
    merged = agg.merge(human_sel, on="question_id", how="inner")
    merged = merged.dropna(subset=["human_difficulty", "mean_p"])

    if merged.empty:
        raise RuntimeError("After merge there are no rows with both human_difficulty and mean_p.")

    os.makedirs(args.out, exist_ok=True)
    merged.to_csv(os.path.join(args.out, "human_machine_merge.csv"), index=False)

    corr = rank_corr(merged["human_difficulty"], 1.0 - merged["mean_p"])

    with open(os.path.join(args.out, "correlation.txt"), "w") as f:
        f.write(f"n_rows: {len(merged)}\n")
        for k, v in corr.items():
            f.write(f"{k}: {v:.6f}\n")

    print("Wrote ->", os.path.join(args.out, "human_machine_merge.csv"))
    print("Wrote ->", os.path.join(args.out, "correlation.txt"))


def main():
    ap = argparse.ArgumentParser(description="Human–Machine Difficulty Alignment pipeline")
    sub = ap.add_subparsers()

    sp = sub.add_parser("demo", help="run a tiny demo with a dummy scorer")
    sp.add_argument("--rounds", type=int, default=3)
    sp.set_defaults(func=cmd_demo)

    sp = sub.add_parser("score", help="score a dataset with a specified model client")
    sp.add_argument("--dataset", required=True, choices=["race", "eedi"])
    sp.add_argument("--split", default="test")
    sp.add_argument("--model", default="dummy", help="e.g. dummy | openai:gpt-4o")
    sp.add_argument("--rounds", type=int, default=5)
    sp.add_argument("--out", required=True)
    sp.set_defaults(func=cmd_score)

    sp = sub.add_parser("datamap", help="compute Data Map summary/regions from scores.csv")
    sp.add_argument("--inp", required=True)
    sp.add_argument("--out", required=True)
    sp.add_argument("--quantile", action="store_true", help="use quantile-based adaptive thresholds")
    sp.set_defaults(func=cmd_datamap)

    sp = sub.add_parser("calibrate", help="temperature scaling + ECE/Brier on scores.csv")
    sp.add_argument("--inp", required=True)
    sp.add_argument("--out", required=True)
    sp.set_defaults(func=cmd_calibrate)

    sp = sub.add_parser("align", help="compute human–machine alignment correlations")
    sp.add_argument("--scores", required=True, help="path to scores.csv")
    sp.add_argument("--human", required=True, help="path to question_summary_task34.csv (or similar)")
    sp.add_argument("--out", required=True)
    sp.set_defaults(func=cmd_align)

    args = ap.parse_args()
    if hasattr(args, "func"):
        args.func(args)
    else:
        ap.print_help()


if __name__ == "__main__":
    main()