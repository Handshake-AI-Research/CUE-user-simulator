"""Shared conversion for Hugging Face chat corpora."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from typing import Any, Callable

from cue_training.preprocessing.raw.conversation_utils import convert_rows


@dataclass(frozen=True)
class ChatCorpusConfig:
    repo: str
    dataset_name: str
    default_splits: tuple[str, ...]
    id_fn: Callable[[Mapping[str, Any], int], str]
    default_exclude_toxic: bool = False


def convert_chat_rows(
    rows: Iterable[Mapping[str, Any]],
    config: ChatCorpusConfig,
    *,
    split: str = "train",
    **kwargs: Any,
) -> list[dict[str, Any]]:
    return convert_rows(
        rows,
        dataset_name=config.dataset_name,
        id_fn=config.id_fn,
        split=split,
        **kwargs,
    )
