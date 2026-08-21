"""Normalize LMSYS-Chat-1M rows."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from typing import Any

from cue_training.preprocessing.raw.process_hf_chat import ChatCorpusConfig, convert_chat_rows

DEFAULT_REPO = "lmsys/lmsys-chat-1m"
DATASET_NAME = "lmsys-chat-1m"


def _conversation_id(row: Mapping[str, Any], index: int) -> str:
    return str(row.get("conversation_id") or f"lmsys-{index}")


CONFIG = ChatCorpusConfig(DEFAULT_REPO, DATASET_NAME, ("train",), _conversation_id)


def convert_rows(
    rows: Iterable[Mapping[str, Any]],
    *,
    split: str = "train",
    **kwargs: Any,
) -> list[dict[str, Any]]:
    return convert_chat_rows(rows, CONFIG, split=split, **kwargs)
