"""Normalize chat formats into CUE sessions (turn lists, not token ids)."""

from __future__ import annotations

from typing import Any

from cue_hf.session_preprocess import normalize_session_preprocess, preprocess_turns

_ROLES = {"user", "assistant", "system"}


class CueProcessor:
    """Turn common chat shapes into ``[[{"role", "content"}, ...], ...]``.

    This produces *structure*, not ``input_ids``: CUE tokenizes each turn separately
    inside the encoder, so there is no session-level tokenizer to call here.

    Accepted inputs:

    - a session: ``[{"role": ..., "content": ...}, ...]`` (OpenAI/HF chat messages,
      including list-valued ``content`` parts, which are joined into text)
    - a batch: a list of such sessions
    - a bare string: treated as one user turn (convenient but lossy)
    """

    def __init__(self, *, session_preprocess: str = "full") -> None:
        self.session_preprocess = normalize_session_preprocess(session_preprocess)

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path: Any, **kwargs: Any) -> CueProcessor:
        """Adopt the ``session_preprocess`` default recorded in a model's config."""

        from cue_hf.configuration_cue import CueConfig

        hub = {key: kwargs.pop(key) for key in ("revision", "token") if key in kwargs}
        config = CueConfig.from_pretrained(pretrained_model_name_or_path, **hub)
        kwargs.setdefault("session_preprocess", config.session_preprocess)
        return cls(**kwargs)

    def __call__(
        self,
        messages: Any,
        *,
        session_preprocess: str | None = None,
    ) -> list[list[dict[str, str]]]:
        mode = (
            self.session_preprocess if session_preprocess is None else normalize_session_preprocess(session_preprocess)
        )
        sessions = [
            [turn for turn in (_normalize_turn(raw) for raw in session) if turn] for session in _as_sessions(messages)
        ]
        empty = [i for i, session in enumerate(sessions) if not session]
        if empty:
            raise ValueError(f"sessions {empty} have no usable user/assistant turns")
        return [preprocess_turns(session, mode) for session in sessions]


def _as_sessions(messages: Any) -> list[list[Any]]:
    if isinstance(messages, str):
        return [[{"role": "user", "content": messages}]]
    if not isinstance(messages, (list, tuple)) or not messages:
        raise ValueError("messages must be a nonempty string, session, or list of sessions")
    first = messages[0]
    if isinstance(first, (list, tuple)):
        return [list(session) for session in messages]
    return [list(messages)]


def _normalize_turn(raw: Any) -> dict[str, str] | None:
    if isinstance(raw, str):
        return {"role": "user", "content": raw.strip()} if raw.strip() else None
    if not isinstance(raw, dict):
        return None
    role = str(raw.get("role") or "").strip().lower()
    if role not in _ROLES:
        return None
    content = _content_text(raw.get("content"))
    return {"role": role, "content": content} if content else None


def _content_text(content: Any) -> str:
    if isinstance(content, str):
        return content.strip()
    if isinstance(content, (list, tuple)):
        # OpenAI multimodal content parts: keep the text ones.
        parts = [str(part.get("text") or "") if isinstance(part, dict) else str(part) for part in content]
        return " ".join(p.strip() for p in parts if p.strip()).strip()
    return "" if content is None else str(content).strip()
