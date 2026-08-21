"""decoder refinement examples and deterministic leakage-audit controls."""

from __future__ import annotations

import hashlib
import random
from collections.abc import Iterable, Iterator
from pathlib import Path
from typing import Any

from cue_training.data.streaming import RoundRobinStreamingDataset, iter_jsonl

Turn = dict[str, Any]
Example = dict[str, Any]


def _copy_turns(turns: Iterable[Turn]) -> list[Turn]:
    return [
        {"role": turn.get("role"), "content": str(turn.get("content") or ""), **{
            key: value for key, value in turn.items() if key not in {"role", "content"}
        }}
        for turn in turns
    ]


def extract_task_text(record: dict[str, Any]) -> str:
    """Return an explicit task field, falling back to the first user turn."""

    for container in (record, record.get("metadata") or {}):
        if not isinstance(container, dict):
            continue
        for key in ("task_description", "intent", "task"):
            value = container.get(key)
            if isinstance(value, dict):
                value = value.get("description") or value.get("text") or value.get("name")
            text = str(value or "").strip()
            if text:
                return text
    for turn in record.get("turns") or []:
        if turn.get("role") == "user":
            return str(turn.get("content") or "").strip()
    return ""


def target_masked_session(turns: Iterable[Turn], target_index: int) -> list[Turn]:
    """Copy a session and hide only the selected target's text."""

    copied = _copy_turns(turns)
    if not 0 <= target_index < len(copied):
        raise IndexError("target_index out of range")
    copied[target_index]["content"] = "<TARGET_USER_TURN>"
    return copied


def future_shuffled_session(
    turns: Iterable[Turn],
    target_index: int,
    *,
    seed: int = 0,
    session_id: str = "",
) -> list[Turn]:
    """Shuffle future content within each role using a stable local seed."""

    copied = _copy_turns(turns)
    if not 0 <= target_index < len(copied):
        raise IndexError("target_index out of range")
    digest = hashlib.sha256(f"{seed}:{session_id}:{target_index}".encode()).digest()
    rng = random.Random(int.from_bytes(digest[:8], "big"))
    by_role: dict[str, list[int]] = {}
    for index in range(target_index + 1, len(copied)):
        by_role.setdefault(str(copied[index].get("role")), []).append(index)
    for indices in by_role.values():
        contents = [copied[index]["content"] for index in indices]
        rng.shuffle(contents)
        for index, content in zip(indices, contents, strict=True):
            copied[index]["content"] = content
    return copied


def examples_from_record(record: dict[str, Any], *, seed: int = 0) -> Iterator[Example]:
    """Yield one example for every user target that has prior context."""

    session_id = str(record.get("session_id") or record.get("id") or "")
    turns = _copy_turns(record.get("turns") or [])
    task_text = extract_task_text(record)
    for target_index, turn in enumerate(turns):
        if target_index < 1 or turn.get("role") != "user":
            continue
        yield {
            "session_id": session_id,
            "full_session": _copy_turns(turns),
            "history": _copy_turns(turns[:target_index]),
            "human_turn": str(turn.get("content") or ""),
            "task_text": task_text,
            "target_index": target_index,
            "target_masked_full_session": target_masked_session(turns, target_index),
            "future_shuffled_full_session": future_shuffled_session(
                turns, target_index, seed=seed, session_id=session_id
            ),
        }


def iter_refinement_examples(
    source: str | Path | Iterable[dict[str, Any]],
    *,
    split: str = "train",
    seed: int = 0,
    per_source_cap: int = 10_000,
) -> Iterator[Example]:
    """Read a profiled streaming dataset, JSONL path, or record iterable."""

    if isinstance(source, (str, Path)):
        path = Path(source)
        if path.is_file():
            records: Iterable[dict[str, Any]] = iter_jsonl(path)
        else:
            records = RoundRobinStreamingDataset(
                path,
                split=split,
                seed=seed,
                per_source_cap=per_source_cap,
                shuffle_buffer=0,
                require_manual=True,
            )
    else:
        records = source
    for record in records:
        yield from examples_from_record(record, seed=seed)
