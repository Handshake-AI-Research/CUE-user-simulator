"""Normalize LoCoMo into one record per speaker session."""

from __future__ import annotations

import re
from collections.abc import Mapping
from typing import Any

from cue_training.preprocessing.raw.schemas import canonical_record

DATASET_NAME = "locomo"
_SESSION_RE = re.compile(r"^session_(\d+)$")


def _turn_text(turn: Mapping[str, Any]) -> str:
    for key in ("text", "clean_text", "utterance", "content"):
        if turn.get(key):
            return str(turn[key]).strip()
    return ""


def convert_sample(
    sample: Mapping[str, Any],
    index: int,
    *,
    split: str = "eval",
) -> list[dict[str, Any]]:
    conversation = sample.get("conversation")
    if not isinstance(conversation, Mapping):
        return []
    sample_id = str(sample.get("sample_id") or f"locomo-{index}")
    numbers = sorted(
        int(match.group(1))
        for key in conversation
        if (match := _SESSION_RE.match(str(key)))
    )
    records = []
    for target in (conversation.get("speaker_a"), conversation.get("speaker_b")):
        if not target:
            continue
        target = str(target)
        for number in numbers:
            session_id = f"{sample_id}:session_{number}:{target}"
            turns: list[dict[str, str]] = []
            pending_context: str | None = None
            for entry in conversation.get(f"session_{number}", []) or []:
                if not isinstance(entry, Mapping):
                    continue
                text = _turn_text(entry)
                if not text:
                    continue
                if str(entry.get("speaker") or "").strip() == target:
                    if pending_context:
                        turns.append({"role": "assistant", "content": pending_context})
                    turns.append({"role": "user", "content": text})
                    pending_context = None
                else:
                    pending_context = text
            if not any(turn["role"] == "user" for turn in turns):
                continue
            records.append(
                canonical_record(
                    session_id,
                    DATASET_NAME,
                    split,
                    turns,
                    metadata={
                        "user_id": f"{sample_id}:{target}",
                        "speaker": target,
                        "session_index": number,
                        "timestamp": conversation.get(f"session_{number}_date_time"),
                    },
                )
            )
    return records


def convert_dataset(
    samples: list[Mapping[str, Any]],
    *,
    split: str = "eval",
) -> list[dict[str, Any]]:
    records = []
    for index, sample in enumerate(samples):
        if isinstance(sample, Mapping):
            records.extend(convert_sample(sample, index, split=split))
    return records
