"""Hugging Face Inference Providers / InferenceClient backend."""

from __future__ import annotations

import os
from typing import TYPE_CHECKING, Any

from huggingface_hub import InferenceClient

if TYPE_CHECKING:
    from collections.abc import Sequence


class HuggingFaceSimulator:
    """Call a chat model through ``huggingface_hub.InferenceClient``."""

    def __init__(
        self,
        model: str,
        *,
        token: str | None = None,
        provider: str | None = None,
    ) -> None:
        kwargs: dict[str, Any] = {
            "model": model,
            "token": token or os.environ.get("HF_TOKEN"),
        }
        if provider:
            kwargs["provider"] = provider
        self.model = model
        self._client = InferenceClient(**kwargs)

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
        reply = self._client.chat_completion(
            messages=payload,
            max_tokens=max_tokens,
            temperature=temperature,
        )
        return (reply.choices[0].message.content or "").strip()
