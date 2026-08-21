"""Thin LiteLLM chat wrapper shared by agents and prompt-based simulators."""

from __future__ import annotations

import asyncio
import json
import os
import random
import re
import threading
import time
from collections.abc import Awaitable, Callable, Iterator, Sequence
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Any, Iterable, TypeVar

from cue_training.runlog.log import progress

# Quiet LiteLLM's per-error "Give Feedback / Get Help" stderr banner (pure noise -- the real
# exception is surfaced by the callers' try/except). Set before litellm is imported anywhere.
os.environ.setdefault("LITELLM_LOG", "ERROR")

Message = dict[str, str]


def _quiet_litellm(litellm: Any) -> None:
    """Suppress LiteLLM's debug/feedback banner (idempotent, cheap)."""
    try:
        litellm.suppress_debug_info = True
    except Exception:  # noqa: BLE001
        pass

_T = TypeVar("_T")
_R = TypeVar("_R")

_JSON_RE = re.compile(r"\{.*\}|\[.*\]", re.DOTALL)

# Active cost accumulator for ``chat`` (shared across ThreadPool workers).
_ACTIVE_COST: "CostTracker | None" = None
_ACTIVE_COST_GUARD = threading.Lock()


@dataclass
class CostTracker:
    """Thread-safe running total of LiteLLM token usage / USD for a tagging pass."""

    n_calls: int = 0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    cost_usd: float = 0.0
    models: set[str] = field(default_factory=set)
    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False)

    def add_response(self, response: Any, *, model: str) -> None:
        import litellm

        usage = getattr(response, "usage", None)
        prompt = int(getattr(usage, "prompt_tokens", 0) or 0) if usage is not None else 0
        completion = int(getattr(usage, "completion_tokens", 0) or 0) if usage is not None else 0
        cost = 0.0
        try:
            cost = float(litellm.completion_cost(completion_response=response) or 0.0)
        except Exception:  # noqa: BLE001
            # Fall back to published per-token rates when litellm cannot price the response.
            try:
                info = litellm.model_cost.get(model) or litellm.model_cost.get(model.split("/")[-1]) or {}
                cost = prompt * float(info.get("input_cost_per_token") or 0.0) + completion * float(
                    info.get("output_cost_per_token") or 0.0
                )
            except Exception:  # noqa: BLE001
                cost = 0.0
        with self._lock:
            self.n_calls += 1
            self.prompt_tokens += prompt
            self.completion_tokens += completion
            self.cost_usd += cost
            if model:
                self.models.add(model)

    def summary(self) -> dict[str, Any]:
        with self._lock:
            return {
                "n_calls": self.n_calls,
                "prompt_tokens": self.prompt_tokens,
                "completion_tokens": self.completion_tokens,
                "cost_usd": round(self.cost_usd, 6),
                "models": sorted(self.models),
            }

    def format_line(self, *, label: str = "fma") -> str:
        s = self.summary()
        models = ",".join(s["models"]) or "?"
        return (
            f"[{label}] cost ~${s['cost_usd']:.4f} "
            f"({s['n_calls']} calls, {s['prompt_tokens']:,} in / {s['completion_tokens']:,} out, "
            f"model={models})"
        )


@contextmanager
def accumulate_chat_cost() -> Iterator[CostTracker]:
    """Accumulate ``chat`` usage/cost for the duration of the block (thread-safe)."""

    global _ACTIVE_COST
    tracker = CostTracker()
    with _ACTIVE_COST_GUARD:
        prev = _ACTIVE_COST
        _ACTIVE_COST = tracker
    try:
        yield tracker
    finally:
        with _ACTIVE_COST_GUARD:
            _ACTIVE_COST = prev


def _record_active_cost(response: Any, *, model: str) -> None:
    tracker = _ACTIVE_COST
    if tracker is not None:
        tracker.add_response(response, model=model)


def parse_json_block(text: str) -> Any:
    """Best-effort extraction of a JSON object/array from an LLM response."""

    text = text.strip()
    if text.startswith("```"):
        text = text.strip("`")
        text = re.sub(r"^(json|JSON)\s*", "", text).strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    match = _JSON_RE.search(text)
    if match:
        try:
            return json.loads(match.group(0))
        except json.JSONDecodeError:
            return None
    return None


def _is_reasoning_chat_model(model: str) -> bool:
    m = (model or "").lower()
    return any(tag in m for tag in ("gpt-5", "o1", "o3", "o4"))


def _has_thinking_budget(model: str) -> bool:
    """Gemini 2.5+/3.x spend thinking tokens from the same output budget as visible text."""

    m = (model or "").lower()
    if "gemini" not in m:
        return False
    return any(tag in m for tag in ("gemini-2.5", "gemini-3", "flash-lite", "flash-preview"))


def _is_anthropic_model(model: str) -> bool:
    m = (model or "").lower()
    return "anthropic" in m or "claude" in m


def _requires_temperature_one(model: str) -> bool:
    """Models that reject any temperature other than 1.

    Claude Fable / Opus 4.7+ and gpt-5* (incl. Terra) only accept the default.
    """

    m = (model or "").lower()
    if _is_reasoning_chat_model(model):
        return True
    return any(
        tag in m
        for tag in (
            "fable",
            "opus-4-7",
            "opus_4_7",
            "opus-4.7",
            "opus_4.7",
            "opus-4-8",
            "opus_4_8",
            "opus-4.8",
            "opus_4.8",
        )
    )


def _needs_output_headroom(model: str) -> bool:
    """True when a tiny max_tokens cap can starve visible content (hidden thinking)."""

    return (
        _is_reasoning_chat_model(model)
        or _has_thinking_budget(model)
        or _requires_temperature_one(model)  # Fable / Opus 4.7+ burn thinking into max_tokens
    )


def _effective_max_tokens(model: str, max_tokens: int) -> int:
    """Raise the completion budget for thinking models so visible JSON is not starved."""

    n = max(1, int(max_tokens))
    if _needs_output_headroom(model):
        return max(n, 8192)
    return n


def _completion_token_kwargs(model: str, max_tokens: int, kwargs: dict[str, Any]) -> dict[str, Any]:
    """Build LiteLLM token kwargs. Reasoning models need ``max_completion_tokens`` (budget
    includes hidden thinking); sending only ``max_tokens`` often yields empty ``content``."""

    out = dict(kwargs)
    if _is_reasoning_chat_model(model):
        out.setdefault("max_completion_tokens", int(max_tokens))
        out.pop("max_tokens", None)
    else:
        out.setdefault("max_tokens", int(max_tokens))
    # Anthropic rejects temperature and top_p together; callers always send temperature.
    if _is_anthropic_model(model):
        out.pop("top_p", None)
    return out


def _coerce_content(content: Any) -> str:
    """Normalize chat message content (str | list of parts | None) to plain text."""

    if content is None:
        return ""
    if isinstance(content, str):
        return content.strip()
    if isinstance(content, list):
        parts: list[str] = []
        for item in content:
            if isinstance(item, str) and item.strip():
                parts.append(item.strip())
            elif isinstance(item, dict):
                typ = str(item.get("type") or "")
                if typ in {"", "text", "output_text", "input_text"}:
                    text = item.get("text") or item.get("content") or ""
                    if isinstance(text, str) and text.strip():
                        parts.append(text.strip())
                elif typ in {"refusal"} and item.get("refusal"):
                    parts.append(str(item["refusal"]).strip())
        return "\n".join(parts).strip()
    return str(content).strip()


def _message_text(response: Any, *, model: str) -> str:
    choice = response.choices[0]
    msg = choice.message
    raw_content = getattr(msg, "content", None)
    content = _coerce_content(raw_content)
    if content:
        return content
    # Some providers put visible text beside reasoning; try common fallbacks.
    for attr in ("reasoning_content", "refusal"):
        alt = _coerce_content(getattr(msg, attr, None))
        if alt:
            return alt
    # LiteLLM / OpenAI sometimes stash parts under provider-specific fields.
    extra = getattr(msg, "provider_specific_fields", None) or {}
    if isinstance(extra, dict):
        for key in ("reasoning_content", "refusal", "text"):
            alt = _coerce_content(extra.get(key))
            if alt:
                return alt
    fr = getattr(choice, "finish_reason", None)
    usage = getattr(response, "usage", None)
    details = getattr(usage, "completion_tokens_details", None) if usage is not None else None
    reasoning_toks = getattr(details, "reasoning_tokens", None) if details is not None else None
    completion_toks = getattr(usage, "completion_tokens", None) if usage is not None else None
    # Dump enough to tell whitespace-only vs true empty vs filtered.
    try:
        dumped = msg.model_dump() if hasattr(msg, "model_dump") else getattr(msg, "__dict__", {})
        dump_preview = repr(dumped)[:400]
    except Exception:  # noqa: BLE001
        dump_preview = repr(msg)[:400]
    hint = (
        "thinking likely exhausted the budget — raise max_completion_tokens or lower reasoning_effort"
        if fr == "length"
        else "blank/whitespace or filtered output (completion_tokens>0 with empty content is often '\\n'*n)"
    )
    print(
        f"[llm] empty content from {model} (finish_reason={fr}, "
        f"completion_tokens={completion_toks}, reasoning_tokens={reasoning_toks}, "
        f"content_type={type(raw_content).__name__}, raw_repr={raw_content!r}). {hint}. "
        f"message={dump_preview}",
        flush=True,
    )
    return ""


def _default_request_timeout() -> float:
    """Per-request wall clock for a single LLM call (LLM_REQUEST_TIMEOUT to override)."""

    # Kept well under a human's patience: a stalled gateway must surface as a retry
    # before it looks like a hang, or the run gets killed before it can self-heal.
    try:
        return max(1.0, float(os.environ.get("LLM_REQUEST_TIMEOUT", "120")))
    except (TypeError, ValueError):
        return 120.0


def _is_transient_llm_error(exc: BaseException) -> bool:
    """True for provider blips that are safe to retry (disconnects, 429/5xx, timeouts)."""

    name = type(exc).__name__
    return any(
        tag in name
        for tag in (
            "Timeout",
            "RateLimitError",
            "RateLimit",
            "APIConnectionError",
            "APIConnection",
            "ServiceUnavailableError",
            "ServiceUnavailable",
            "InternalServerError",
            "InternalServer",
        )
    )


def chat(
    messages: list[Message],
    *,
    model: str,
    api_key_env: str | None = "OPENAI_API_KEY",
    api_base: str | None = None,
    temperature: float = 0.7,
    max_tokens: int = 1024,
    retries_on_empty: int = 1,
    retries_transient: int = 6,
    **kwargs: Any,
) -> str:
    """Return the assistant message content for a LiteLLM chat completion."""

    import litellm

    _quiet_litellm(litellm)
    # Without this a stalled socket blocks its worker forever: a high-concurrency
    # pass looks like a hang once every thread is parked on a dead connection.
    # A timeout raises litellm.Timeout, which _is_transient_llm_error retries.
    kwargs.setdefault("timeout", _default_request_timeout())
    api_key = os.getenv(api_key_env) if api_key_env else None
    max_tokens = _effective_max_tokens(model, max_tokens)
    call_kwargs = _completion_token_kwargs(model, max_tokens, kwargs)
    # gpt-5* + reasoning_effort only accepts temperature=1; omit when effort is set and value is default-ish.
    # Claude Fable 5 / Opus 4.7+ likewise reject any temperature other than 1.
    temp = temperature
    if _requires_temperature_one(model) or (
        _is_reasoning_chat_model(model) and call_kwargs.get("reasoning_effort")
    ):
        temp = 1.0
    attempts = max(1, int(retries_on_empty) + 1)
    transient_tries = max(1, int(retries_transient))
    last = ""
    for attempt in range(attempts):
        response = None
        for transient_attempt in range(transient_tries):
            try:
                response = litellm.completion(
                    model=model,
                    messages=messages,
                    temperature=temp,
                    api_key=api_key,
                    api_base=api_base,
                    **call_kwargs,
                )
                break
            except Exception as exc:  # noqa: BLE001
                if not _is_transient_llm_error(exc) or transient_attempt + 1 >= transient_tries:
                    raise
                delay = min(60.0, (2**transient_attempt) + random.uniform(0.0, 1.0))
                print(
                    f"[llm] {type(exc).__name__} from {model} "
                    f"(attempt {transient_attempt + 1}/{transient_tries}); "
                    f"retry in {delay:.1f}s...",
                    flush=True,
                )
                time.sleep(delay)
        assert response is not None
        _record_active_cost(response, model=model)
        last = _message_text(response, model=model)
        if last:
            return last
        if attempt + 1 < attempts:
            print(f"[llm] retrying empty {model} response ({attempt + 2}/{attempts})...", flush=True)
    return last


async def abatch_map(
    items: Sequence[_T],
    coro_fn: Callable[[_T], Awaitable[_R]],
    *,
    concurrency: int = 10,
    desc: str | None = None,
) -> list[_R]:
    """Run ``coro_fn`` over ``items`` with bounded concurrency and a progress bar.

    Processes up to ``concurrency`` items at a time on a single event loop (true async
    I/O via ``acompletion``), so data-prep LLM calls parallelize without a thread pool.
    Results are returned in input order; ``concurrency`` bounds in-flight requests.
    """

    results: list[_R] = []
    step = max(1, int(concurrency))
    bar = progress(total=len(items), desc=desc, unit="call", rank_zero=False)
    for start in range(0, len(items), step):
        chunk = items[start : start + step]
        results.extend(await asyncio.gather(*[coro_fn(item) for item in chunk]))
        bar.update(len(chunk))
    bar.close()
    return results


def astream_map(
    items: Iterable[_T],
    coro_fn: Callable[[_T], Awaitable[_R]],
    *,
    concurrency: int = 10,
    desc: str | None = None,
    total: int | None = None,
) -> Iterator[_R]:
    """Stream ``coro_fn`` over an iterable with bounded concurrency, yielding in order.

    Unlike :func:`abatch_map`, ``items`` need not be a materialized sequence -- it is
    consumed lazily in ``concurrency``-sized chunks, so a round-robin record stream can
    be labeled without loading the whole dataset into memory. A single event loop is
    reused across chunks (one loop avoids litellm async-client/loop-rebinding issues).
    """

    bar = progress(total=total, desc=desc, unit="call", rank_zero=False)
    loop = asyncio.new_event_loop()

    async def _gather(chunk: list[_T]) -> list[_R]:
        return await asyncio.gather(*[coro_fn(item) for item in chunk])

    try:
        chunk: list[_T] = []
        step = max(1, int(concurrency))
        for item in items:
            chunk.append(item)
            if len(chunk) >= step:
                for result in loop.run_until_complete(_gather(chunk)):
                    yield result
                    bar.update(1)
                chunk = []
        if chunk:
            for result in loop.run_until_complete(_gather(chunk)):
                yield result
                bar.update(1)
    finally:
        loop.close()
        bar.close()


async def achat(
    messages: list[Message],
    *,
    model: str,
    api_key_env: str | None = "OPENAI_API_KEY",
    api_base: str | None = None,
    temperature: float = 0.7,
    max_tokens: int = 1024,
    retries_on_empty: int = 1,
    **kwargs: Any,
) -> str:
    """Async counterpart of :func:`chat` using ``litellm.acompletion``.

    Lets the rollout driver run many episodes concurrently on a single event loop
    (true async I/O) rather than one OS thread per concurrent episode.
    """

    import litellm

    _quiet_litellm(litellm)
    kwargs.setdefault("timeout", _default_request_timeout())
    api_key = os.getenv(api_key_env) if api_key_env else None
    call_kwargs = _completion_token_kwargs(model, max_tokens, kwargs)
    temp = temperature
    if _is_reasoning_chat_model(model) and call_kwargs.get("reasoning_effort"):
        temp = 1.0
    attempts = max(1, int(retries_on_empty) + 1)
    last = ""
    for attempt in range(attempts):
        response = await litellm.acompletion(
            model=model,
            messages=messages,
            temperature=temp,
            api_key=api_key,
            api_base=api_base,
            **call_kwargs,
        )
        last = _message_text(response, model=model)
        if last:
            return last
        if attempt + 1 < attempts:
            print(f"[llm] retrying empty {model} response ({attempt + 2}/{attempts})...", flush=True)
    return last
