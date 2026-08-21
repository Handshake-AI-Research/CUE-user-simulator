"""Read normalized MirrorBench records (rollout writing lives in the env eval now)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

Turn = dict[str, str]


def load_normalized(path: Path, limit: int | None = None) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with open(path, encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            records.append(json.loads(line))
            if limit is not None and len(records) >= limit:
                break
    return records


def real_conversation(record: dict[str, Any]) -> list[Turn]:
    return list(record.get("real_conversation") or [])


def count_user_turns(conversation: list[Turn]) -> int:
    return sum(1 for turn in conversation if turn.get("role") == "user")


def num_user_turns(record: dict[str, Any]) -> int:
    """How many user turns the synthetic rollout should produce."""

    meta = record.get("metadata") or {}
    explicit = meta.get("n_user_turns")
    if isinstance(explicit, int) and explicit > 0:
        return explicit
    return count_user_turns(real_conversation(record))


def task_text(record: dict[str, Any]) -> str:
    return str(record.get("task_description") or record.get("intent") or "")
