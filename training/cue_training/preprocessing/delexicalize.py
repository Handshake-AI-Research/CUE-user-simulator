"""Typed delexicalization for persona-manual targets only."""

from __future__ import annotations

import re
from typing import Any

_REPLACERS = [
    (re.compile(r"https?://\S+"), "<URL>"),
    (re.compile(r"\b\d{1,2}[/-]\d{1,2}[/-]\d{2,4}\b"), "<NUM>"),
    (re.compile(r"\$\s*\d+(?:\.\d+)?"), "$<NUM>"),
    (re.compile(r"\b\d+(?:\.\d+)?\b"), "<NUM>"),
    (re.compile(r"`[^`]+`"), "<CODE>"),
    (re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b"), "<ENTITY>"),
    (re.compile(r"\b(?:order|account|user)\s*ids?\b", re.I), "<TERM>"),
    (re.compile(r"\b(?:username|password|pin|membership)\b", re.I), "<TERM>"),
    (re.compile(r"\b(?:bronze|silver|gold|platinum|guest)\s*(?:member(?:ship)?)?\b", re.I), "<TERM>"),
    (re.compile(r"\bguess\s+shirts?\b", re.I), "<TERM>"),
    (re.compile(r"\b(?:shipping\s+)?labels?\b", re.I), "<TERM>"),
    (re.compile(r"\"[^\"]{2,}\""), '"<ENTITY>"'),
]


def delexicalize_text(text: str) -> str:
    out = text
    for pattern, repl in _REPLACERS:
        out = pattern.sub(repl, out)
    return out


def delexicalize_manual(manual: dict[str, Any]) -> dict[str, Any]:
    commands = []
    for entry in manual.get("commands", []):
        commands.append(
            {
                **entry,
                "text": delexicalize_text(str(entry.get("text") or "")),
                "examples": [delexicalize_text(x) for x in entry.get("examples") or []],
            }
        )
    return {"schema_version": manual.get("schema_version", 1), "commands": commands}
