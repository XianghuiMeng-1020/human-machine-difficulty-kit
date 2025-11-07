from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Mapping, Optional, Sequence


__all__ = [
    "ScoreResult",
    "ModelClient",
    "normalize_probs",
    "validate_option_keys",
    "argmax_key",
]


LETTERS: Sequence[str] = ("A", "B", "C", "D")


@dataclass
class ScoreResult:
    """
    Result of scoring a multiple-choice question.

    Attributes
    ----------
    question_id : str
        Stable identifier for the item.
    chosen : str
        Final option selected by the model, one of "A","B","C","D".
    probs : Dict[str, float]
        Normalized probability distribution over options A..D.
    p_correct : Optional[float]
        Probability assigned to the gold option if known;
        callers can fill this as probs[gold] when available.
    meta : Optional[dict]
        Optional metadata (e.g., raw logits, latency, model config).
    """
    question_id: str
    chosen: str
    probs: Dict[str, float]
    p_correct: Optional[float] = None
    meta: Optional[dict] = None


def validate_option_keys(options: Mapping[str, str]) -> None:
    """
    Validate that the options mapping contains exactly A, B, C, D.
    """
    keys = set(options.keys())
    expected = set(LETTERS)
    if keys != expected:
        raise ValueError(f"options must have keys {sorted(expected)}, got {sorted(keys)}")


def normalize_probs(probs: Mapping[str, float]) -> Dict[str, float]:
    """
    Normalize and validate a probability distribution over A..D.

    - Requires all keys A,B,C,D to be present.
    - Clips negatives to zero.
    - Renormalizes to sum to 1. If all values are non-positive, uses uniform.

    Returns
    -------
    Dict[str, float] : normalized probabilities.
    """
    keys = set(probs.keys())
    expected = set(LETTERS)
    if keys != expected:
        raise ValueError(f"probs must have keys {sorted(expected)}, got {sorted(keys)}")

    vals = {k: (float(v) if v is not None else 0.0) for k, v in probs.items()}
    vals = {k: (v if v > 0.0 else 0.0) for k, v in vals.items()}
    s = sum(vals.values())
    if s <= 0.0:
        return {k: 1.0 / len(LETTERS) for k in LETTERS}
    return {k: vals[k] / s for k in LETTERS}


def argmax_key(probs: Mapping[str, float]) -> str:
    """
    Return the option key with maximum probability.
    """
    best_k = None
    best_v = float("-inf")
    for k, v in probs.items():
        fv = float(v)
        if fv > best_v:
            best_v = fv
            best_k = k
    # best_k cannot be None if probs is non-empty
    return str(best_k)


class ModelClient:
    """
    Base interface for model scoring clients.

    Implementations must override `score_mcq` to return a ScoreResult with:
    - probs: normalized distribution over A..D (use `normalize_probs`).
    - chosen: argmax over probs (use `argmax_key`).
    """

    name: str = "abstract"

    def score_mcq(self, prompt: str, options: Mapping[str, str]) -> ScoreResult:
        """
        Score a multiple-choice question.

        Parameters
        ----------
        prompt : str
            Formatted prompt including passage/question text.
        options : Mapping[str, str]
            Mapping from option letter to option text, with keys A..D.

        Returns
        -------
        ScoreResult
            Scoring result with normalized probs and chosen letter.

        Notes
        -----
        This method must be implemented by subclasses. The base implementation
        raises a clear error if called directly.
        """
        raise RuntimeError(
            "ModelClient.score_mcq must be implemented by a concrete subclass."
        )