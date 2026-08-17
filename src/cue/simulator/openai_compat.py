"""OpenAI-compatible chat backends (OpenAI, vLLM, OpenRouter, Azure, …)."""

from __future__ import annotations

import importlib
import os
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from collections.abc import Sequence

_OPENAI_REQUIRED = "OpenAICompatSimulator needs the openai package: pip install 'cue-runtime[openai]'"


class OpenAICompatSimulator:
    """Call any endpoint that speaks the OpenAI Chat Completions API."""

    def __init__(
        self,
        model: str,
        *,
        api_key: str | None = None,
        base_url: str | None = None,
    ) -> None:
        try:
            module = importlib.import_module("openai")
        except ImportError as exc:
            raise ImportError(_OPENAI_REQUIRED) from exc
        openai = module.OpenAI
        self.model = model
        self._client: Any = openai(
            api_key=api_key or os.environ.get("OPENAI_API_KEY"),
            base_url=base_url or os.environ.get("OPENAI_BASE_URL"),
        )

    def chat(
        self,
        messages: Sequence[dict[str, str]],
        *,
        system: str | None = None,
        temperature: float = 0.8,
        max_tokens: int = 160,
    ) -> str:
        payload = list(messages)
        if system:
            payload = [{"role": "system", "content": system}, *payload]
        reply = self._client.chat.completions.create(
            model=self.model,
            messages=payload,
            temperature=temperature,
            max_tokens=max_tokens,
        )
        return (reply.choices[0].message.content or "").strip()
