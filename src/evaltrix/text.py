"""Small text helpers shared by the built-in evals."""

from __future__ import annotations

import json
import re
from typing import Any

_ABBREV = {"e.g.", "i.e.", "etc.", "vs.", "mr.", "mrs.", "ms.", "dr.", "st.", "no.", "approx.", "u.s.", "inc.", "ltd."}
_BOUNDARY = re.compile(r"(?<=[.!?])[\"')\]]*\s+(?=[\"'(\[]?[A-Z0-9])")
_BULLET = re.compile(r"^\s*(?:[-*•]|\d+[.)])\s+")


def split_sentences(text: str) -> list[str]:
    """Split prose into sentences; list items and lines count as separate sentences.

    Decimals ("3.5"), and common abbreviations ("e.g.", "Dr.") don't end a sentence.
    """
    out: list[str] = []
    for line in text.splitlines():
        line = _BULLET.sub("", line).strip()
        if not line:
            continue
        start = 0
        for m in _BOUNDARY.finditer(line):
            candidate = line[start:m.start()].strip()
            last_word = candidate.rsplit(" ", 1)[-1].lower()
            if last_word in _ABBREV:
                continue
            if candidate:
                out.append(candidate)
            start = m.end()
        tail = line[start:].strip()
        if tail:
            out.append(tail)
    return out


def truncate(text: Any, limit: int) -> str:
    s = text if isinstance(text, str) else json.dumps(text, default=str)
    return s if len(s) <= limit else s[: limit - 1] + "…"
