"""Normalize WildChat rows."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from typing import Any

from cue_training.preprocessing.raw.process_hf_chat import ChatCorpusConfig, convert_chat_rows

DEFAULT_REPO = "allenai/WildChat-1M"
DATASET_NAME = "wildchat"


def _conversation_id(row: Mapping[str, Any], index: int) -> str:
    value = row.get("conversation_hash") or row.get("conversation_id")
    return f"{value}-{index}" if value not in (None, "") else f"wildchat-{index}"


CONFIG = ChatCorpusConfig(DEFAULT_REPO, DATASET_NAME, ("train",), _conversation_id, True)


def convert_rows(
    rows: Iterable[Mapping[str, Any]],
    *,
    split: str = "train",
    exclude_toxic: bool = True,
    **kwargs: Any,
) -> list[dict[str, Any]]:
    return convert_chat_rows(
        rows,
        CONFIG,
        split=split,
        exclude_toxic=exclude_toxic,
        **kwargs,
    )
