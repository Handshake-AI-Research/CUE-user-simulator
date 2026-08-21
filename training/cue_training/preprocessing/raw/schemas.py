"""Canonical raw records consumed by unified data annotation."""

from __future__ import annotations

from typing import Any, NotRequired, TypedDict

from cue_training.data.schema import validate_record


class RawTurn(TypedDict):
    role: str
    content: str
    turn_id: str


class RawRecord(TypedDict):
    id: str
    session_id: str
    dataset: str
    split: str
    turns: list[RawTurn]
    metadata: NotRequired[dict[str, Any]]


def canonical_record(
    session_id: Any,
    dataset: str,
    split: str,
    turns: list[dict[str, Any]],
    *,
    metadata: dict[str, Any] | None = None,
) -> RawRecord:
    """Validate and return the minimal data annotation raw-record shape."""

    record = validate_record(
        {
            "id": str(session_id),
            "dataset": dataset,
            "split": split,
            "turns": turns,
            "metadata": metadata or {},
        }
    )
    result: RawRecord = {
        "id": record["id"],
        "session_id": record["session_id"],
        "dataset": record["dataset"],
        "split": record["split"],
        "turns": record["turns"],
    }
    if metadata:
        result["metadata"] = metadata
    return result
