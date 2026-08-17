"""Turn OpenAI-style chat messages into CUE session turns."""

from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path
from typing import TYPE_CHECKING, Any, cast

if TYPE_CHECKING:
    from collections.abc import Sequence

_ROLE_ALIASES = {
    "human": "user",
    "ai": "assistant",
    "bot": "assistant",
    "model": "assistant",
}
_EMPTY_CONVERSATION = "conversation is empty"
_NO_TURNS = "no user/assistant turns found in conversation"
_EMPTY_TRANSCRIPT = "transcript has no turns"


def normalize_messages(
    messages: Sequence[Mapping[str, Any]] | Sequence[str] | str,
) -> list[dict[str, str]]:
    """Accept common chat formats and return ``[{role, content}, ...]``.

    Supported inputs:

    - OpenAI / HF chat list: ``[{"role": "user", "content": "..."}, ...]``
    - Role-prefixed transcript string (``user: ...`` / ``assistant: ...`` per line)
    - Bare string list (treated as alternating user / assistant, starting with user)
    """

    if isinstance(messages, str):
        return _from_transcript(messages)
    if not messages:
        raise ValueError(_EMPTY_CONVERSATION)
    first = messages[0]
    if isinstance(first, str):
        text_messages = cast("Sequence[str]", messages)
        return [
            {"role": "user" if i % 2 == 0 else "assistant", "content": text.strip()}
            for i, text in enumerate(text_messages)
            if text.strip()
        ]
    turns: list[dict[str, str]] = []
    mapped_messages = cast("Sequence[Mapping[str, Any]]", messages)
    for item in mapped_messages:
        role = str(item.get("role") or "").strip().lower()
        role = _ROLE_ALIASES.get(role, role)
        content = item.get("content")
        if isinstance(content, list):
            # Multimodal OpenAI content parts → keep text only.
            content = " ".join(
                str(part.get("text") or "").strip()
                for part in content
                if isinstance(part, Mapping) and part.get("type", "text") == "text"
            )
        text = str(content or "").strip()
        if role not in {"user", "assistant", "system"} or not text:
            continue
        turns.append({"role": role, "content": text})
    if not turns:
        raise ValueError(_NO_TURNS)
    return turns


def _from_transcript(text: str) -> list[dict[str, str]]:
    turns: list[dict[str, str]] = []
    for line in text.splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        role, _, content = stripped.partition(":")
        role_key = role.strip().lower()
        role_key = _ROLE_ALIASES.get(role_key, role_key)
        if role_key in {"user", "assistant", "system"} and content.strip():
            turns.append({"role": role_key, "content": content.strip()})
        elif turns:
            turns[-1]["content"] += " " + stripped
        else:
            turns.append({"role": "user", "content": stripped})
    if not turns:
        raise ValueError(_EMPTY_TRANSCRIPT)
    return turns


def load_conversation(path: str) -> list[dict[str, str]]:
    """Load a conversation from JSON, JSONL, or a role-prefixed text file."""

    raw = Path(path).read_text(encoding="utf-8")
    stripped = raw.strip()
    if not stripped:
        message = f"{path} is empty"
        raise ValueError(message)
    if stripped[0] in "[{":
        payload = json.loads(stripped)
        if isinstance(payload, dict) and "messages" in payload:
            payload = payload["messages"]
        if isinstance(payload, list):
            return normalize_messages(payload)
    # JSONL: one message object per line.
    lines = [line for line in stripped.splitlines() if line.strip()]
    if lines and lines[0].lstrip().startswith("{"):
        try:
            return normalize_messages([json.loads(line) for line in lines])
        except json.JSONDecodeError:
            pass
    return normalize_messages(raw)
