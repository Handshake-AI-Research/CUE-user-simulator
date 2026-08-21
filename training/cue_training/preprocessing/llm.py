"""Thin LLM completion helper (litellm) with retries and context truncation."""

from __future__ import annotations

import json
import os
import re
from typing import Any

from cue_training.utils.log import warn

TAG = "llm"

# Rough chars/token for local truncation (conservative for code-heavy tau2 transcripts).
_CHARS_PER_TOKEN = 3.5


def estimate_tokens(text: str) -> int:
    return max(1, int(len(text) / _CHARS_PER_TOKEN))


def truncate_messages(
    messages: list[dict[str, str]],
    *,
    max_input_tokens: int,
    max_tokens: int = 128,
) -> list[dict[str, str]]:
    """Keep system + newest turns so prompt + max_tokens fits the context window."""

    budget = max(256, max_input_tokens - max_tokens - 32)
    if not messages:
        return messages
    system = [{**m, "content": m.get("content") or ""} for m in messages if m.get("role") == "system"]
    rest = [m for m in messages if m.get("role") != "system"]
    used = sum(estimate_tokens(m["content"]) for m in system)
    # Manuals / retrieved style examples can alone exceed the local vLLM window; keep the tail.
    if used > budget and system:
        overflow = used - budget
        for i, msg in enumerate(system):
            content = msg["content"]
            cost = estimate_tokens(content)
            if cost <= 0:
                continue
            drop = min(len(content), int(overflow * _CHARS_PER_TOKEN) + 64)
            if drop <= 0:
                break
            trimmed = content[drop:]
            if not trimmed.strip():
                trimmed = content[-max(200, len(content) // 4) :]
            system[i] = {**msg, "content": trimmed}
            overflow -= estimate_tokens(content) - estimate_tokens(trimmed)
            if overflow <= 0:
                break
        used = sum(estimate_tokens(m["content"]) for m in system)
    kept: list[dict[str, str]] = []
    for msg in reversed(rest):
        cost = estimate_tokens(msg.get("content") or "") + 4
        if kept and used + cost > budget:
            break
        # Always try to keep at least the last user/assistant turn; truncate its body if huge.
        content = msg.get("content") or ""
        if cost > budget - used and not kept:
            max_chars = max(200, int((budget - used) * _CHARS_PER_TOKEN))
            content = content[-max_chars:]
            cost = estimate_tokens(content) + 4
        if used + cost > budget and kept:
            break
        kept.append({**msg, "content": content})
        used += cost
    kept.reverse()
    return system + kept


def _strip_fences(text: str) -> str:
    text = text.strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*", "", text, count=1, flags=re.IGNORECASE)
        text = re.sub(r"\s*```$", "", text, count=1)
    return text.strip()


def _extract_json_object(text: str) -> dict[str, Any]:
    text = _strip_fences(text)
    try:
        data = json.loads(text)
        if isinstance(data, dict):
            return data
    except json.JSONDecodeError:
        pass
    start = text.find("{")
    end = text.rfind("}")
    if start < 0 or end <= start:
        raise json.JSONDecodeError("No JSON object found", text, 0)
    blob = text[start : end + 1]
    # Common model glitches: trailing commas.
    blob = re.sub(r",\s*([}\]])", r"\1", blob)
    data = json.loads(blob)
    if not isinstance(data, dict):
        raise json.JSONDecodeError("JSON root is not an object", blob, 0)
    return data


def complete(
    *,
    model: str,
    messages: list[dict[str, str]],
    temperature: float = 0.2,
    max_tokens: int = 1500,
    api_base: str | None = None,
    api_key: str | None = None,
    response_format: dict[str, Any] | None = None,
    max_input_tokens: int | None = None,
) -> str:
    try:
        import litellm
        from litellm import completion
        from tenacity import retry, stop_after_attempt, wait_exponential
    except ImportError as exc:
        raise ImportError(
            "data annotation LLM calls require optional deps: pip/uv install 'cue[llm]'"
        ) from exc

    # OpenRouter ids (meta-llama/llama-3.1-8b-instruct) miss litellm's bare-name provider
    # lookup in get_max_tokens, which prints a "Provider List" banner on every single call.
    litellm.suppress_debug_info = True

    if max_input_tokens is not None:
        messages = truncate_messages(
            messages, max_input_tokens=max_input_tokens, max_tokens=max_tokens
        )

    kwargs: dict[str, Any] = {
        "model": model,
        "messages": messages,
        "temperature": temperature,
        "max_tokens": max_tokens,
    }
    if api_base:
        kwargs["api_base"] = api_base
    if api_key:
        kwargs["api_key"] = api_key
    elif api_base and "OPENAI_API_KEY" not in os.environ:
        kwargs["api_key"] = os.environ.get("HOSTED_VLLM_API_KEY", "EMPTY")
    if response_format is not None:
        kwargs["response_format"] = response_format

    # Longer/backoff-heavier retry so a sustained org rate limit (e.g. Anthropic input-tokens/min)
    # is ridden out across the per-minute window rather than exhausting attempts and dropping the
    # candidate; max wait > 60s lets the limit reset before the final try.
    @retry(wait=wait_exponential(multiplier=2, min=2, max=90), stop=stop_after_attempt(8), reraise=True)
    def _call() -> str:
        try:
            resp = completion(**kwargs)
        except Exception as exc:  # noqa: BLE001
            warn(TAG, f"{model} failed: {exc}")
            raise
        return (resp.choices[0].message.content or "").strip()

    return _call()


def complete_json(
    *,
    model: str,
    messages: list[dict[str, str]],
    temperature: float = 0.2,
    max_tokens: int = 2500,
    api_base: str | None = None,
    api_key: str | None = None,
    response_format: dict[str, Any] | None = None,
    max_input_tokens: int | None = None,
    repair_attempts: int = 2,
) -> dict[str, Any]:
    """Complete and parse JSON, with fence stripping and one repair pass on failure."""

    text = complete(
        model=model,
        messages=messages,
        temperature=temperature,
        max_tokens=max_tokens,
        api_base=api_base,
        api_key=api_key,
        response_format=response_format,
        max_input_tokens=max_input_tokens,
    )
    try:
        return _extract_json_object(text)
    except json.JSONDecodeError as first_exc:
        last_exc: Exception = first_exc
        for attempt in range(repair_attempts):
            warn(TAG, f"{model} JSON parse failed ({first_exc}); repair attempt {attempt + 1}")
            repair_messages = [
                {
                    "role": "system",
                    "content": (
                        "Repair the following into a single valid JSON object. "
                        "Output ONLY the JSON, no markdown."
                    ),
                },
                {"role": "user", "content": text[:12000]},
            ]
            try:
                text = complete(
                    model=model,
                    messages=repair_messages,
                    temperature=0.0,
                    max_tokens=max_tokens,
                    api_base=api_base,
                    api_key=api_key,
                    response_format={"type": "json_object"},
                    max_input_tokens=max_input_tokens,
                )
                return _extract_json_object(text)
            except Exception as exc:  # noqa: BLE001
                last_exc = exc
        raise last_exc from first_exc
