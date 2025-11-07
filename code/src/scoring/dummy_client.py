from __future__ import annotations

import numpy as np

from .interface import (
    ModelClient,
    ScoreResult,
    normalize_probs,
    argmax_key,
    LETTERS,
)


class DummyClient(ModelClient):
    """
    Deterministic dummy scorer that returns a random-looking distribution over A..D.

    Useful for exercising the pipeline without calling any external APIs.
    """

    name: str = "dummy"

    def __init__(self, seed: int = 42) -> None:
        self.rng = np.random.default_rng(seed)

    def score_mcq(self, prompt: str, options: dict[str, str]) -> ScoreResult:
        # Generate pseudo-random logits and softmax to probabilities
        z = self.rng.normal(loc=0.0, scale=1.0, size=4)
        p = np.exp(z - np.max(z))  # softmax stabilization
        p = p / p.sum()

        probs = {LETTERS[i]: float(p[i]) for i in range(4)}
        probs = normalize_probs(probs)
        chosen = argmax_key(probs)

        return ScoreResult(question_id="", chosen=chosen, probs=probs)