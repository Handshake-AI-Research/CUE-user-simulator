"""Session turn preprocess for CUE encode.

Writing assistants often return a short meta lead-in plus a long document draft.
``strip_document`` keeps the meta and drops the draft body so fusion is less
dominated by task-document tokens.
"""

from __future__ import annotations

import re
from typing import Any, Literal

SessionPreprocessMode = Literal["full", "strip_document", "user_only"]

SESSION_PREPROCESS_MODES: tuple[str, ...] = ("full", "strip_document", "user_only")

DRAFT_STUB = "[draft]"
_META_MAX_WORDS = 80

_PREAMBLE_RE = re.compile(
    r"^(sure|here|of course|absolutely|i(['’]?ve| have)|let me|great|okay|ok|"
    r"thanks|here['’]?s|here is|i can|i['’]?ll|i will|i['’]?d|happy to|"
    r"no problem|got it|revised|updated|done|perfect|alright|all right|"
    r"sounds good|makes sense|i('?d| would) be)",
    re.IGNORECASE,
)

_DRAFT_MARKER_RE = re.compile(r"(?im)^(subject:|dear |hi |hello |to:|from:|#{1,3} |\*\*|---+|blog|post:)")

_META_HINT_RE = re.compile(
    r"(?i)\b("
    r"i (can|could|would|have|['’]?ve)|here['’]?s|here is|let me|revised|"
    r"version|draft|updated|changes?|tone|formal"
    r")\b"
)


def normalize_session_preprocess(mode: str | None) -> SessionPreprocessMode:
    value = str(mode or "full").strip().lower() or "full"
    if value not in SESSION_PREPROCESS_MODES:
        raise ValueError(f"session_preprocess must be one of {SESSION_PREPROCESS_MODES}, got {mode!r}")
    return value  # type: ignore[return-value]


def _word_count(text: str) -> int:
    return len(text.split())


def _cap_words(text: str, max_words: int) -> str:
    words = text.split()
    if len(words) <= max_words:
        return text.strip()
    return " ".join(words[:max_words]).strip()


def _has_preamble(text: str) -> bool:
    return bool(_PREAMBLE_RE.search(text.lstrip()[:400]))


def _looks_draftish(text: str) -> bool:
    words = _word_count(text)
    qmarks = text.count("?")
    return bool(_DRAFT_MARKER_RE.search(text)) or (words > 80 and qmarks <= 1)


def strip_assistant_document(
    text: str,
    *,
    meta_max_words: int = _META_MAX_WORDS,
) -> str:
    """Keep short assistant meta; drop document body (or stub draft-only turns)."""

    content = (text or "").strip()
    if not content:
        return ""

    n_words = _word_count(content)
    qmarks = content.count("?")
    draftish = _looks_draftish(content)

    # Short clarify / discuss-only — keep as-is.
    if n_words < 60 and qmarks >= 1 and not draftish:
        return content
    if n_words < 80 and _has_preamble(content) and not draftish:
        return content
    if qmarks >= 2 and n_words < 150 and not draftish:
        return content

    # Preamble + body via blank-line split.
    blank = re.search(r"\n\s*\n", content)
    if blank:
        head = content[: blank.start()].strip()
        tail = content[blank.end() :].strip()
        if head and _has_preamble(head) and (_word_count(tail) > 40 or _looks_draftish(tail)):
            return _cap_words(head, meta_max_words)

    # Lead-in before draft markers (Subject:/Dear / markdown...).
    marker = _DRAFT_MARKER_RE.search(content)
    if marker and marker.start() > 0:
        head = content[: marker.start()].strip()
        if head and (_has_preamble(head) or _word_count(head) <= meta_max_words):
            return _cap_words(head, meta_max_words)

    # Weak meta then long draft: keep first paragraph if it looks meta-ish.
    if n_words > 100:
        meta_hits = len(_META_HINT_RE.findall(content[:300]))
        if meta_hits >= 2 and blank:
            head = content[: blank.start()].strip()
            if head:
                return _cap_words(head, meta_max_words)

    # Draft-only / long document with no usable meta.
    if draftish or n_words > 80:
        return DRAFT_STUB

    return content


def preprocess_turns(
    turns: list[dict[str, Any]],
    mode: str | None = "full",
) -> list[dict[str, Any]]:
    """Return a shallow-copied turn list transformed for encode."""

    normalized = normalize_session_preprocess(mode)
    if normalized == "full":
        return list(turns)

    if normalized == "user_only":
        return [dict(turn) for turn in turns if str(turn.get("role") or "") == "user"]

    # strip_document
    out: list[dict[str, Any]] = []
    for turn in turns:
        role = str(turn.get("role") or "")
        copied = dict(turn)
        if role in {"assistant", "system"}:
            copied["content"] = strip_assistant_document(str(turn.get("content") or ""))
        out.append(copied)
    return out
