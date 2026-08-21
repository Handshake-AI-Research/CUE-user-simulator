"""Map original SimArena / PRISM assistant IDs to LiteLLM models.

Providers are restricted to keys the CUE rollout host is assumed to have:
OpenAI, Anthropic, Google/Gemini, and OpenRouter (for Llama / Mistral / Phi /
Cohere / other open models). No direct ``mistral/`` or ``cohere/`` provider ids.

SimArena human logs store short names on each episode (``model`` / ``assistant_model``).
PRISM stores the upstream ``model_name`` (often a HF id) on enriched ``normalized.jsonl``.
"""

from __future__ import annotations

import threading
from collections import Counter
from typing import Iterable

# SimulatorArena annotation short names → LiteLLM ids (exact when still served).
SIMARENA_ASSISTANT_MAP: dict[str, str] = {
    "gpt-4o": "gpt-4o",
    "gpt-4o-mini": "gpt-4o-mini",
    "gpt-4-turbo": "gpt-4-turbo",
    "claude-3-5-sonnet-20240620": "anthropic/claude-sonnet-4-5",
    "mistral-large-2407": "openrouter/mistralai/mistral-large-2407",
    "mistral-large-latest": "openrouter/mistralai/mistral-large",
    "llama-3-1-8b": "openrouter/meta-llama/llama-3.1-8b-instruct",
    "llama-3-1-70b": "openrouter/meta-llama/llama-3.1-70b-instruct",
    "phi-3-medium": "openrouter/microsoft/phi-4",
    "phi-3-small": "openrouter/microsoft/phi-4",
}

# PRISM upstream model_name values (+ short paper aliases) → closest allowed providers.
PRISM_ASSISTANT_MAP: dict[str, str] = {
    # OpenAI (exact / closest)
    "gpt-4": "gpt-4",
    "gpt-4-turbo": "gpt-4-turbo",
    "gpt-4-1106-preview": "gpt-4-turbo",
    "gpt-3.5-turbo": "gpt-3.5-turbo",
    # Anthropic
    "claude-2": "anthropic/claude-sonnet-4-5",
    "claude-2.1": "anthropic/claude-sonnet-4-5",
    "claude-instant-1": "anthropic/claude-haiku-4-5",
    # Google PaLM → Gemini
    "palm-2": "gemini/gemini-2.0-flash",
    "models/chat-bison-001": "gemini/gemini-2.0-flash",
    "google/flan-t5-xxl": "gemini/gemini-2.0-flash",
    "flan-t5-xxl": "gemini/gemini-2.0-flash",
    # Cohere via OpenRouter
    "command": "openrouter/cohere/command-r-plus-08-2024",
    "command-light": "openrouter/cohere/command-r-08-2024",
    "command-nightly": "openrouter/cohere/command-r-plus-08-2024",
    # Aleph Alpha luminous → Cohere-sized openrouter stand-in
    "luminous-supreme-control": "openrouter/cohere/command-r-plus-08-2024",
    "luminous-extended-control": "openrouter/cohere/command-r-08-2024",
    # Llama-2 (paper short + HF ids) → Llama-3.1
    "llama-2-7b-chat": "openrouter/meta-llama/llama-3.1-8b-instruct",
    "llama-2-13b-chat": "openrouter/meta-llama/llama-3.1-8b-instruct",
    "llama-2-70b-chat": "openrouter/meta-llama/llama-3.1-70b-instruct",
    "meta-llama/Llama-2-7b-chat-hf": "openrouter/meta-llama/llama-3.1-8b-instruct",
    "meta-llama/Llama-2-13b-chat-hf": "openrouter/meta-llama/llama-3.1-8b-instruct",
    "meta-llama/Llama-2-70b-chat-hf": "openrouter/meta-llama/llama-3.1-70b-instruct",
    # Mistral / Zephyr / Falcon / Guanaco / Pythia
    "mistral-7b-instruct": "openrouter/mistralai/ministral-8b-2512",
    "mistralai/Mistral-7B-Instruct-v0.1": "openrouter/mistralai/ministral-8b-2512",
    "zephyr-7b-beta": "openrouter/mistralai/ministral-8b-2512",
    "HuggingFaceH4/zephyr-7b-beta": "openrouter/mistralai/ministral-8b-2512",
    "guanaco-33b": "openrouter/meta-llama/llama-3.1-70b-instruct",
    "timdettmers/guanaco-33b-merged": "openrouter/meta-llama/llama-3.1-70b-instruct",
    "falcon-7b-instruct": "openrouter/meta-llama/llama-3.1-8b-instruct",
    "tiiuae/falcon-7b-instruct": "openrouter/meta-llama/llama-3.1-8b-instruct",
    "pythia-12b": "openrouter/microsoft/phi-4",
    "OpenAssistant/oasst-sft-4-pythia-12b-epoch-3.5": "openrouter/microsoft/phi-4",
}

# When PRISM episode metadata lacks a chosen model, use a model that was in the
# original PRISM pool and is still available via OpenAI.
PRISM_DEFAULT_ASSISTANT = "gpt-4-turbo"

_ALLOWED_KEY_ENVS = frozenset({
    "OPENAI_API_KEY",
    "ANTHROPIC_API_KEY",
    "GOOGLE_API_KEY",
    "OPENROUTER_API_KEY",
})


def api_key_env_for_model(model: str) -> str:
    """Env var for the four supported providers (OpenAI / Anthropic / Gemini / OpenRouter)."""

    m = (model or "").lower()
    if m.startswith("openrouter/"):
        return "OPENROUTER_API_KEY"
    if m.startswith("anthropic/") or m.startswith("claude"):
        return "ANTHROPIC_API_KEY"
    if m.startswith("gemini/") or m.startswith("google/"):
        return "GOOGLE_API_KEY"
    if m.startswith(("gpt-", "o1", "o3", "o4", "chatgpt", "openai/")):
        return "OPENAI_API_KEY"
    return "OPENROUTER_API_KEY"


def _ensure_allowed(model: str) -> str:
    env = api_key_env_for_model(model)
    if env not in _ALLOWED_KEY_ENVS:
        raise ValueError(f"assistant model {model!r} needs unsupported key env {env}")
    low = model.lower()
    if low.startswith("mistral/") or low.startswith("cohere/"):
        raise ValueError(
            f"assistant model {model!r} uses a provider outside "
            f"OpenAI/Anthropic/Gemini/OpenRouter; route it via openrouter/..."
        )
    return model


def resolve_simarena_assistant(dataset_model: str | None, *, fallback: str | None = None) -> str:
    """Resolve a SimArena annotation ``model`` string to a LiteLLM id."""

    key = str(dataset_model or "").strip()
    if not key:
        if fallback:
            return _ensure_allowed(fallback)
        raise ValueError("empty SimArena assistant model")
    if key in SIMARENA_ASSISTANT_MAP:
        return _ensure_allowed(SIMARENA_ASSISTANT_MAP[key])
    if "/" in key or key.startswith(("gpt-", "claude-", "gemini", "o1", "o3")):
        return _ensure_allowed(key)
    if fallback:
        return _ensure_allowed(fallback)
    raise KeyError(
        f"unknown SimArena assistant {key!r}; add it to SIMARENA_ASSISTANT_MAP "
        f"or pass a full LiteLLM id"
    )


class DatasetAssistantRouter:
    """Route dataset-matched assistants around models a provider refuses to serve.

    Some original assistants are retired or blocked by the account's OpenRouter data
    policy. A model is marked dead on its first failure and every later call routes to
    the dataset's most frequent still-working model, so episodes finish instead of
    truncating mid-conversation.
    """

    def __init__(self, models: Iterable[str]) -> None:
        counts = Counter(str(m) for m in models if m)
        self._ranked = [model for model, _ in counts.most_common()]
        self._dead: set[str] = set()
        self._lock = threading.Lock()

    def route(self, model: str) -> str:
        """The model to call: ``model`` unless it is dead, else the most popular live one."""

        with self._lock:
            if model not in self._dead:
                return model
            for candidate in self._ranked:
                if candidate not in self._dead:
                    return candidate
        return model

    def mark_dead(self, model: str) -> str | None:
        """Record a failing model; returns the replacement, or None when none is left."""

        from cue_training.runlog.log import warn

        with self._lock:
            first = model not in self._dead
            self._dead.add(model)
            alt = next((c for c in self._ranked if c not in self._dead), None)
        if first:
            warn(
                "dataset-assistants",
                f"assistant {model!r} unavailable; routing later turns to {alt!r}"
                if alt
                else f"assistant {model!r} unavailable and no dataset model is left",
            )
        return alt


def resolve_prism_assistant(dataset_model: str | None, *, fallback: str | None = None) -> str:
    """Resolve a PRISM model name to a LiteLLM id (or the PRISM default stand-in)."""

    key = str(dataset_model or "").strip()
    if not key:
        return _ensure_allowed(fallback or PRISM_DEFAULT_ASSISTANT)
    if key in PRISM_ASSISTANT_MAP:
        return _ensure_allowed(PRISM_ASSISTANT_MAP[key])
    if key.startswith(("gpt-", "claude-", "gemini", "o1", "o3")):
        return _ensure_allowed(key)
    if key.startswith(("openrouter/", "anthropic/", "gemini/", "openai/")):
        return _ensure_allowed(key)
    return _ensure_allowed(fallback or PRISM_DEFAULT_ASSISTANT)
