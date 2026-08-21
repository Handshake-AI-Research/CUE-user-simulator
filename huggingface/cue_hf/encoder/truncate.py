"""Text helpers for CUE encoder context windows."""

from __future__ import annotations

from typing import Any


def truncate_text_head_tail(
    text: str,
    tokenizer: Any,
    *,
    max_length: int,
    head_tokens: int = 128,
    tail_tokens: int = 128,
    prefix: str = "search_document: ",
) -> str:
    """Keep the start and end of a long turn so truncation does not drop the tail.

    The system encoder wraps text as ``prefix + text`` and truncates to ``max_length``
    tokens (HF default: keep the *prefix*). For long assistant drafts that loses the
    ending. When ``prefix+text`` exceeds ``max_length``, keep the first ``head_tokens``
    and last ``tail_tokens`` of the body (shrunk to fit the remaining budget).
    """

    if not text or max_length <= 0:
        return text
    # verbose=False: these encodes only measure/slice, and the caller re-tokenizes the
    # result with truncation, so transformers' "longer than the specified maximum" warning
    # fires on every long turn without anything over-length ever reaching the model.
    full_ids = tokenizer.encode(f"{prefix}{text}", add_special_tokens=False, verbose=False)
    if len(full_ids) <= max_length:
        return text
    prefix_ids = tokenizer.encode(prefix, add_special_tokens=False)
    budget = max(1, int(max_length) - len(prefix_ids))
    body_ids = tokenizer.encode(text, add_special_tokens=False, verbose=False)
    if len(body_ids) <= budget:
        return text
    head = min(int(head_tokens), budget)
    tail = min(int(tail_tokens), max(0, budget - head))
    if head + tail > budget:
        head = budget // 2
        tail = budget - head
    kept = body_ids[:head] + (body_ids[-tail:] if tail else [])
    return tokenizer.decode(kept, skip_special_tokens=True)
