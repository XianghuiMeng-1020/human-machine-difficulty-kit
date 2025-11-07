# src/scoring/prompts.py

from __future__ import annotations
from typing import Dict, Optional


def _fmt_options(options: Dict[str, str] | None) -> str:
    if not options:
        # still show the labels so the model knows the valid set
        return "Options:\nA) \nB) \nC) \nD) "
    # keep canonical A-D order
    lines = ["Options:"]
    for k in ["A", "B", "C", "D"]:
        v = options.get(k, "")
        v = "" if v is None else str(v)
        lines.append(f"{k}) {v}")
    return "\n".join(lines)


def build_prompt(
    passage: Optional[str],
    question: Optional[str],
    options: Optional[Dict[str, str]],
) -> str:
    """
    Builds a single string instruction. It is image-aware:
    - If no passage/question text is provided (typical for Eedi Task 3/4),
      we explicitly tell the model that an image of a 4-choice MCQ is attached.
    - Always enforce: respond with only one of A/B/C/D (single uppercase letter).
    """
    passage = (passage or "").strip()
    question = (question or "").strip()

    blocks: list[str] = []

    # If there is no textual content, instruct the model to read the image.
    if not passage and not question:
        blocks.append(
            "You will receive an image of a multiple-choice math question with four options "
            "labeled A, B, C, and D. Carefully read all text and diagrams in the image and "
            "determine the single best answer."
        )
    else:
        if passage:
            blocks.append("Passage:\n" + passage)
        # it's fine if question is empty; we still show the heading for consistency
        blocks.append("Question:\n" + question)

    blocks.append(_fmt_options(options))

    # Final instruction: output format must be exactly one letter
    blocks.append(
        "Answer format: output only a single uppercase letter from {A,B,C,D} with no extra text."
    )

    return "\n\n".join(blocks)