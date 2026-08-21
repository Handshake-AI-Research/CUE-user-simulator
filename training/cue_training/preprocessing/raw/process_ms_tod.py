"""Normalize MS-TOD persona sessions."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from cue_training.preprocessing.raw.conversation_utils import conversation_to_turns
from cue_training.preprocessing.raw.schemas import canonical_record

DATASET_NAME = "ms-tod"
_UTTERANCE_KEYS = ("turns", "utterances", "dialogue", "dialog", "log", "messages")
_SESSION_KEYS = ("sessions", "dialogues", "dialogs", "history")


def _first(mapping: Mapping[str, Any], keys: Sequence[str]) -> Any:
    for key in keys:
        if mapping.get(key) not in (None, ""):
            return mapping[key]
    return None


def _iter_personas(data: Any):
    if isinstance(data, Mapping):
        for wrapper in ("personas", "data"):
            if isinstance(data.get(wrapper), (list, dict)):
                yield from _iter_personas(data[wrapper])
                return
        if any(key in data for key in _SESSION_KEYS):
            yield str(data.get("persona_id") or data.get("id") or "persona-0"), data
        else:
            for persona_id, value in data.items():
                yield str(persona_id), value
    elif isinstance(data, list):
        for index, value in enumerate(data):
            persona_id = value.get("persona_id") if isinstance(value, Mapping) else None
            yield str(persona_id if persona_id is not None else f"persona-{index}"), value


def convert_dataset(data: Any, *, split: str = "eval") -> list[dict[str, Any]]:
    records = []
    for persona_id, persona in _iter_personas(data):
        sessions = _first(persona, _SESSION_KEYS) if isinstance(persona, Mapping) else persona
        if not isinstance(sessions, list):
            continue
        for order, session in enumerate(sessions):
            if not isinstance(session, Mapping):
                continue
            session_id = str(session.get("session_id") or f"{persona_id}:session_{order}")
            utterances = _first(session, _UTTERANCE_KEYS)
            if not isinstance(utterances, list):
                continue
            turns = conversation_to_turns(utterances, session_id)
            if not any(turn["role"] == "user" for turn in turns):
                continue
            annotation = session.get("annotation")
            annotation = annotation if isinstance(annotation, Mapping) else session
            metadata = {
                "persona_id": persona_id,
                "session_index": order,
                "service": _first(annotation, ("service", "domain")),
                "intent": _first(annotation, ("intent", "active_intent")),
                "reference_dialogue_id": session.get("reference_dialogue_id"),
            }
            records.append(
                canonical_record(
                    session_id,
                    DATASET_NAME,
                    split,
                    turns,
                    metadata={key: value for key, value in metadata.items() if value is not None},
                )
            )
    return records
