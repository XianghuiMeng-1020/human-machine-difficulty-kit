from __future__ import annotations

import json
import os
from typing import Any, Dict, Iterable, Iterator


__all__ = ["read_jsonl", "write_jsonl"]


def _ensure_parent_dir(path: str) -> None:
    """Create parent directory for a file path if it does not exist."""
    parent = os.path.dirname(os.path.abspath(path)) or "."
    os.makedirs(parent, exist_ok=True)


def read_jsonl(path: str) -> Iterator[Dict[str, Any]]:
    """
    Stream records from a JSON Lines file.

    Args:
        path: Path to a .jsonl file.

    Yields:
        Dict parsed from each non-empty line.
    """
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)


def write_jsonl(path: str, records: Iterable[Dict[str, Any]]) -> None:
    """
    Write an iterable of dictionaries to a JSON Lines file.

    Args:
        path: Output .jsonl path. Parent directories are created as needed.
        records: Iterable of dictionaries to serialize.
    """
    _ensure_parent_dir(path)
    with open(path, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")