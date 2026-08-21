"""Content-perturbation views for augmentation consistency."""

from __future__ import annotations

import random
import re
from typing import Any

_PATTERNS = [
    (re.compile(r"https?://\S+"), "<URL>"),
    (re.compile(r"\b\d+(?:\.\d+)?\b"), "<NUM>"),
    (re.compile(r"`[^`]+`"), "<CODE>"),
    (re.compile(r"\"[^\"]+\""), "<ENTITY>"),
    (re.compile(r"'[^']+'"), "<ENTITY>"),
]


def perturb_text(text: str) -> str:
    out = text
    for pattern, repl in _PATTERNS:
        out = pattern.sub(repl, out)
    return out


def perturb_session(turns: list[dict[str, str]]) -> list[dict[str, str]]:
    return [
        {
            "role": t.get("role", ""),
            "content": perturb_text(t.get("content", "")) if t.get("role") == "user" else t.get("content", ""),
        }
        for t in turns
    ]


def dropout_user_turns(
    turns: list[dict[str, str]],
    drop_prob: float,
    *,
    rng: random.Random | None = None,
) -> list[dict[str, str]]:
    """Drop middle user-turn context bundles while retaining the first and last."""

    if drop_prob <= 0:
        return list(turns)
    user_indices = [i for i, turn in enumerate(turns) if turn.get("role") == "user"]
    if len(user_indices) <= 2:
        return list(turns)
    draw = rng.random if rng is not None else random.random
    kept = [
        index
        for position, index in enumerate(user_indices)
        if position in {0, len(user_indices) - 1}
        or draw() >= min(drop_prob, 1.0)
    ]
    out = list(turns[: user_indices[0]])
    kept_set = set(kept)
    for position, user_index in enumerate(user_indices):
        if user_index in kept_set:
            next_user = (
                user_indices[position + 1]
                if position + 1 < len(user_indices)
                else len(turns)
            )
            out.extend(turns[user_index:next_user])
    return out


def select_turns_for_persona(turns: list[dict[str, Any]], max_user_turns: int = 15) -> list[dict[str, Any]]:
    """Keep up to max_user_turns user turns (first, last, evenly spaced) with context."""

    user_indices = [i for i, t in enumerate(turns) if t.get("role") == "user"]
    if len(user_indices) <= max_user_turns:
        return turns
    if max_user_turns <= 1:
        keep = {user_indices[0]}
    elif max_user_turns == 2:
        keep = {user_indices[0], user_indices[-1]}
    else:
        keep = {user_indices[0], user_indices[-1]}
        inner = max_user_turns - 2
        for j in range(inner):
            pos = 1 + int(round(j * (len(user_indices) - 3) / max(1, inner - 1)))
            keep.add(user_indices[min(len(user_indices) - 2, max(1, pos))])
    out: list[dict[str, Any]] = []
    for i, turn in enumerate(turns):
        if turn.get("role") in {"assistant", "system"}:
            # Keep preceding system only if next kept user exists.
            if any(u in keep for u in user_indices if u > i):
                # keep if immediately followed by a kept user
                nxt = next((u for u in user_indices if u > i), None)
                if nxt is not None and nxt in keep and all(
                    turns[k].get("role") != "user" for k in range(i + 1, nxt)
                ):
                    out.append(turn)
            continue
        if i in keep:
            out.append(turn)
    return out
