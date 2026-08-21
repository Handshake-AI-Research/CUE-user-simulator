"""Normalize DialogStudio task-oriented dialogs."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from typing import Any

from cue_training.preprocessing.raw.schemas import canonical_record


def _prompt_text(row: Mapping[str, Any], field: str) -> str:
    value = row.get(field)
    if isinstance(value, list):
        return "\n".join(str(item) for item in value if item)
    return str(value or "").strip()


def _dialog_id(row: Mapping[str, Any]) -> str:
    for key in ("new dialog id", "original dialog id", "dialog index"):
        if row.get(key) not in (None, ""):
            return str(row[key])
    return ""


def convert_dialog(
    row: Mapping[str, Any],
    dataset_name: str,
    prompt_field: str = "prompt",
    *,
    split: str = "train",
    index: int = 0,
) -> dict[str, Any] | None:
    session_id = _dialog_id(row) or f"{dataset_name}-{index}"
    turns: list[dict[str, Any]] = []
    prompt = _prompt_text(row, prompt_field)
    if prompt:
        turns.append({"role": "system", "content": prompt})
    for entry in row.get("log", []) or []:
        if not isinstance(entry, Mapping):
            continue
        user = str(entry.get("user utterance") or "").strip()
        assistant = str(entry.get("system response") or "").strip()
        if user:
            turns.append({"role": "user", "content": user})
        if assistant:
            turns.append({"role": "assistant", "content": assistant})
    if not any(turn["role"] == "user" for turn in turns):
        return None
    return canonical_record(session_id, dataset_name, split, turns)


def convert_rows(
    rows: Iterable[Mapping[str, Any]],
    dataset_name: str,
    prompt_field: str = "prompt",
    max_dialogs: int | None = None,
    *,
    split: str = "train",
) -> list[dict[str, Any]]:
    records = []
    for index, row in enumerate(rows):
        if max_dialogs is not None and len(records) >= max_dialogs:
            break
        record = convert_dialog(
            row,
            dataset_name,
            prompt_field,
            split=split,
            index=index,
        )
        if record is not None:
            records.append(record)
    return records
