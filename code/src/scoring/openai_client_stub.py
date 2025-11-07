from __future__ import annotations

import os
import re
import math
import base64
import mimetypes
from typing import Dict, Optional, List
from dataclasses import dataclass
from openai import OpenAI

# ------------------------------
# Fallback ScoreResult (compat)
# ------------------------------
try:
    from src.scoring.interface import ScoreResult  # preferred
except Exception:
    @dataclass
    class ScoreResult:
        chosen: str
        probs: Dict[str, float]
        question_id: Optional[str] = None


# ------------------------------
# Helpers
# ------------------------------
LETTER_SET = {"A", "B", "C", "D"}


def _norm_letter(text: object) -> Optional[str]:
    """Extract a single A/B/C/D from free-form text."""
    if text is None:
        return None
    s = str(text).strip().upper()

    # Leading single letter (allow trailing punctuation/space)
    m = re.match(r"^\s*([ABCD])(?:\b|[.)\s].*)?$", s)
    if m:
        return m.group(1)

    # Things like "Answer: C"
    m = re.search(r"ANSWER[^A-Z0-9]*([ABCD])\b", s)
    if m:
        return m.group(1)

    # Fallback: last standalone letter token
    toks = re.findall(r"\b([ABCD])\b", s)
    return toks[-1] if toks else None


def _token_to_letter(tok: str) -> Optional[str]:
    """Map a token variant like 'C', 'C.', ' C)' into a letter."""
    if tok is None:
        return None
    t = str(tok).strip().upper()
    m = re.match(r"^([ABCD])(?:[.)])?$", t)
    if m:
        return m.group(1)
    m = re.search(r"[ABCD]", t)
    return m.group(0) if m else None


def _encode_image_to_data_url(path: str) -> Optional[str]:
    if not path or not os.path.exists(path):
        return None
    mime, _ = mimetypes.guess_type(path)
    if mime is None:
        mime = "image/png"
    with open(path, "rb") as f:
        b64 = base64.b64encode(f.read()).decode("utf-8")
    return f"data:{mime};base64,{b64}"


# ------------------------------
# Core client
# ------------------------------
class OpenAIClient:
    """
    Robust MCQ scorer using Chat Completions + logprobs.

    Behavior:
    - Read logprobs from the first generated token that contains A/B/C/D
      (or the very first token if CONF_MODE=first_raw).
    - Normalize probabilities over letters only or over all tokens
      (controlled by NORM_SCOPE).
    - Environment controls:

        USE_IMAGE=1|0          : whether to include an image (as data URL)
        STRICT_LETTER=1|0      : strongly nudge the model to output only a single letter
        LETTER_TOKENS=int      : max_tokens when STRICT_LETTER=1 (default 1)
        TOP_LOGK=int           : top_logprobs K (1..20; default 20)
        CONF_MODE=first_letter | first_raw
                                 - first_letter (default): first token that contains A/B/C/D
                                 - first_raw            : the very first token, fallback to first_letter
        NORM_SCOPE=letters | all
                                 - letters (default): normalize within {A,B,C,D}
                                 - all             : include non-letter mass in the denominator
        GEN_T / TEMPERATURE     : sampling temperature (float, default 1.0)
    """

    def __init__(self, model: str = "gpt-4o"):
        self.model = model
        self.client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))

    # ---- env toggles ----
    def _use_image(self) -> bool:
        return os.getenv("USE_IMAGE", "0") == "1"

    def _strict_letter(self) -> bool:
        return os.getenv("STRICT_LETTER", "1") == "1"

    def _conf_mode(self) -> str:
        m = os.getenv("CONF_MODE", "first_letter").strip().lower()
        return m if m in {"first_letter", "first_raw"} else "first_letter"

    def _top_logk(self) -> int:
        try:
            k = int(os.getenv("TOP_LOGK", "20"))
        except Exception:
            k = 20
        return max(1, min(20, k))  # API requires <= 20

    def _letter_tokens(self) -> int:
        try:
            n = int(os.getenv("LETTER_TOKENS", "1"))
        except Exception:
            n = 1
        return max(1, min(8, n))

    def _temperature(self) -> float:
        raw = os.getenv("GEN_T", os.getenv("TEMPERATURE", "1.0"))
        try:
            t = float(raw)
        except Exception:
            t = 1.0
        if not (0.0 <= t <= 2.0):
            t = 1.0
        return t

    def _norm_scope(self) -> str:
        m = os.getenv("NORM_SCOPE", "letters").strip().lower()
        return m if m in {"letters", "all"} else "letters"

    # ---- main ----
    def score_mcq(
        self,
        prompt: str,
        options: Dict[str, str],
        image_path: Optional[str] = None,
    ) -> ScoreResult:

        # Strong system nudge to return a single letter
        sys_msg = None
        if self._strict_letter():
            sys_msg = {
                "role": "system",
                "content": (
                    "You are a grader. Reply with a single character only: A, B, C, or D. "
                    "No explanation, no punctuation, no spaces."
                ),
            }

        content: List[dict] = [{"type": "text", "text": prompt}]

        if self._use_image() and image_path:
            data_url = _encode_image_to_data_url(image_path)
            if data_url:
                content.append({"type": "image_url", "image_url": {"url": data_url}})

        messages = [{"role": "user", "content": content}]
        if sys_msg:
            messages.insert(0, sys_msg)

        resp = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=self._temperature(),
            max_tokens=(self._letter_tokens() if self._strict_letter() else 2),
            logprobs=True,
            top_logprobs=self._top_logk(),
        )

        # textual output (fallback path)
        text_out = (resp.choices[0].message.content or "").strip()

        # Prepare letter distribution
        probs = {"A": 0.0, "B": 0.0, "C": 0.0, "D": 0.0}

        lp = resp.choices[0].logprobs
        token_entries = getattr(lp, "content", None) if lp else None

        def collect_distribution(entry) -> (Dict[str, float], float):
            """
            Collect unnormalized mass for letters and non-letter mass at this token.
            Includes BOTH the actual token and the top_logprobs alternatives.
            Returns: (letters_mass_dict, other_mass)
            """
            letters = {k: 0.0 for k in LETTER_SET}
            other = 0.0

            # actual token
            act_tok = getattr(entry, "token", "")
            act_lp = getattr(entry, "logprob", None)
            if act_lp is not None:
                act_p = math.exp(float(act_lp))
                letter = _token_to_letter(act_tok)
                if letter in LETTER_SET:
                    letters[letter] += act_p
                else:
                    other += act_p

            # alternatives
            for alt in (getattr(entry, "top_logprobs", []) or []):
                tok = getattr(alt, "token", "")
                logp = getattr(alt, "logprob", None)
                if logp is None:
                    continue
                p = math.exp(float(logp))
                letter = _token_to_letter(tok)
                if letter in LETTER_SET:
                    letters[letter] += p
                else:
                    other += p

            return letters, other

        def normalize_letters(letters: Dict[str, float], other: float) -> Dict[str, float]:
            """
            Apply normalization according to NORM_SCOPE.
            - letters: raw letter masses
            - other  : raw non-letter mass
            """
            scope = self._norm_scope()
            out = {k: 0.0 for k in LETTER_SET}
            sum_letters = sum(letters.values())

            if scope == "letters":
                # Normalize within letters only
                if sum_letters > 0:
                    for k, v in letters.items():
                        out[k] = v / sum_letters
            else:
                # "all": include non-letter mass in the denominator
                denom = sum_letters + other
                if denom > 0:
                    # We keep the mass share of letters vs all tokens
                    # so the sum over A/B/C/D is <= 1.0
                    for k, v in letters.items():
                        out[k] = v / denom
            return out

        got_letter_probs = False
        if token_entries:
            mode = self._conf_mode()

            # helper that tries one entry
            def try_entry(ent) -> Optional[Dict[str, float]]:
                letters, other = collect_distribution(ent)
                # must be a token that actually involves a letter if first_letter mode
                if self._conf_mode() == "first_letter":
                    if _token_to_letter(getattr(ent, "token", "")) not in LETTER_SET:
                        return None
                dist = normalize_letters(letters, other)
                if sum(dist.values()) > 0:
                    return dist
                return None

            if mode == "first_raw":
                # Try the very first generated token
                dist = try_entry(token_entries[0])
                if dist is not None:
                    probs.update(dist)
                    got_letter_probs = True
                # If not useful, fall back to first token that actually contains a letter
                if not got_letter_probs:
                    for ent in token_entries:
                        if _token_to_letter(getattr(ent, "token", "")) in LETTER_SET:
                            dist = try_entry(ent)
                            if dist is not None:
                                probs.update(dist)
                                got_letter_probs = True
                                break
            else:
                # first_letter: scan for the first token that contains A/B/C/D
                for ent in token_entries:
                    if _token_to_letter(getattr(ent, "token", "")) in LETTER_SET:
                        dist = try_entry(ent)
                        if dist is not None:
                            probs.update(dist)
                            got_letter_probs = True
                            break

        # Choose final letter
        if got_letter_probs and sum(probs.values()) > 0:
            chosen = max(probs, key=probs.get)
        else:
            parsed = _norm_letter(text_out)
            chosen = parsed or "A"
            # fallback: one-hot
            probs = {k: (1.0 if k == chosen else 0.0) for k in probs}

        # Return (runner.py does not require question_id here)
        return ScoreResult(question_id="", chosen=chosen, probs=probs)