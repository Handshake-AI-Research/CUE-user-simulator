"""Rollout-record assembly and generated user-turn cleanup."""

import re
from collections.abc import Mapping
from typing import Any

_NEXT_USER_MARKER = "Next user message:\n"
_ROLE_PREFIX = re.compile(
    r"^\s*(?:human|user|assistant|model|customer|agent)\s*(?:[:\-]|\n|$)\s*",
    re.IGNORECASE,
)
STOP_MARKER = "###"


def strip_role_prefix(text: str) -> str:
    cleaned = str(text or "").strip()
    for _ in range(3):
        stripped = _ROLE_PREFIX.sub("", cleaned).strip()
        if stripped == cleaned:
            break
        cleaned = stripped
    return cleaned


def clean_user_message(content: str, *, return_stop: bool = False) -> str | tuple[str, bool]:
    text = str(content or "")
    if _NEXT_USER_MARKER in text:
        text = text.rsplit(_NEXT_USER_MARKER, 1)[1]
    text = strip_role_prefix(text)
    if len(text) >= 2 and text[0] == text[-1] and text[0] in {'"', "'"}:
        text = text[1:-1].strip()
    stop = STOP_MARKER in text
    if stop:
        text = text.split(STOP_MARKER, 1)[0].strip()
    return (text, stop) if return_stop else text


def rollout_record_from_conversation(
    row: Mapping[str, Any], rollout_conversation: list[dict[str, Any]]
) -> dict[str, Any]:
    normalized = []
    for index, turn in enumerate(rollout_conversation):
        metadata: dict[str, Any] = {"turn_index": index}
        raw = turn.get("raw_data") if isinstance(turn, Mapping) else None
        if raw and raw.get("cue_command"):
            metadata["cue_command"] = raw["cue_command"]
        normalized.append(
            {
                "role": "assistant" if str(turn.get("role")) == "assistant" else "user",
                "content": str(turn.get("content") or ""),
                "metadata": metadata,
            }
        )

    episode_id = str(row.get("episode_id") or row.get("task_id") or "")
    dataset = row.get("dataset") or (row.get("metadata") or {}).get("domain")
    metadata = dict(row.get("metadata") or {})
    metadata.setdefault("domain", dataset)
    if row.get("arm"):
        metadata.setdefault("rollout_kind", row["arm"])
    return {
        "dataset": dataset,
        "episode_id": episode_id,
        "task_id": str(row.get("task_id") or episode_id),
        "real_conversation": list(row.get("real_conversation") or []),
        "rollout_conversation": normalized,
        "intent": row.get("intent") or "",
        "task_description": row.get("task_description") or row.get("task") or "",
        "metadata": metadata,
    }
