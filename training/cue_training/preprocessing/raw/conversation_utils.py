"""Pure helpers for normalizing chat conversations."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from datetime import datetime
from typing import Any

from cue_training.preprocessing.raw.schemas import RawRecord, canonical_record

USER_ROLES = frozenset({"user", "human"})
ASSISTANT_ROLES = frozenset({"assistant", "gpt", "bot", "chatbot", "model"})
SYSTEM_ROLES = frozenset({"system"})

_LANGUAGE_ALIASES = {
    "en": "english",
    "eng": "english",
    "english": "english",
    "es": "spanish",
    "pt": "portuguese",
    "zh": "chinese",
    "ru": "russian",
    "de": "german",
    "fr": "french",
    "it": "italian",
    "ja": "japanese",
    "ko": "korean",
}


def normalize_role(role: Any) -> str:
    normalized = str(role or "").strip().lower()
    if normalized in USER_ROLES:
        return "user"
    if normalized in ASSISTANT_ROLES:
        return "assistant"
    if normalized in SYSTEM_ROLES:
        return "system"
    return ""


def message_text(message: Mapping[str, Any]) -> str:
    content = message.get("content")
    if content is None:
        content = message.get("text") or message.get("utterance") or message.get("value")
    if isinstance(content, list):
        parts = [
            str(part.get("text") or part.get("content") or "")
            if isinstance(part, Mapping)
            else str(part or "")
            for part in content
        ]
        return "\n".join(part for part in parts if part).strip()
    return str(content or "").strip()


def conversation_to_turns(
    conversation: Iterable[Mapping[str, Any]],
    session_id: str,
) -> list[dict[str, str]]:
    turns = []
    for message in conversation:
        if not isinstance(message, Mapping):
            continue
        role = normalize_role(message.get("role") or message.get("speaker") or message.get("sender"))
        content = message_text(message)
        if role and content:
            turns.append(
                {
                    "turn_id": str(message.get("turn_id") or f"{session_id}:{role}:{len(turns)}"),
                    "role": role,
                    "content": content,
                }
            )
    return turns


def convert_chat_row(
    row: Mapping[str, Any],
    *,
    dataset_name: str,
    conversation_id: str,
    split: str = "train",
    conversation_field: str = "conversation",
    min_turns: int = 1,
) -> RawRecord | None:
    conversation = row.get(conversation_field)
    if not isinstance(conversation, list):
        return None
    turns = conversation_to_turns(conversation, conversation_id)
    if sum(turn["role"] == "user" for turn in turns) < min_turns:
        return None
    # WildChat hands back a real datetime for `timestamp`, which no JSONL writer
    # downstream can serialize.
    metadata = {
        key: value.isoformat() if isinstance(value := row[key], datetime) else value
        for key in ("language", "model", "timestamp", "conversation_hash", "conversation_id")
        if row.get(key) not in (None, "")
    }
    return canonical_record(
        conversation_id,
        dataset_name,
        split,
        turns,
        metadata=metadata or None,
    )


def canonical_language(value: Any) -> str:
    normalized = str(value or "").strip().lower()
    return _LANGUAGE_ALIASES.get(normalized, normalized)


def passes_filters(
    row: Mapping[str, Any],
    *,
    languages: set[str] | None = None,
    english_only: bool = False,
    exclude_redacted: bool = False,
    exclude_toxic: bool = False,
) -> bool:
    if exclude_redacted and bool(row.get("redacted")):
        return False
    if exclude_toxic and bool(row.get("toxic")):
        return False
    language = canonical_language(row.get("language"))
    if english_only and language and language != "english":
        return False
    targets = {canonical_language(value) for value in languages or set()}
    return not (targets and language and language not in targets)


def convert_rows(
    rows: Iterable[Mapping[str, Any]],
    *,
    dataset_name: str,
    id_fn,
    split: str = "train",
    conversation_field: str = "conversation",
    min_turns: int = 1,
    max_conversations: int | None = None,
    **filter_kwargs: Any,
) -> list[RawRecord]:
    records = []
    for index, row in enumerate(rows):
        if max_conversations is not None and len(records) >= max_conversations:
            break
        if not isinstance(row, Mapping) or not passes_filters(row, **filter_kwargs):
            continue
        record = convert_chat_row(
            row,
            dataset_name=dataset_name,
            conversation_id=id_fn(row, index),
            split=split,
            conversation_field=conversation_field,
            min_turns=min_turns,
        )
        if record is not None:
            records.append(record)
    return records
