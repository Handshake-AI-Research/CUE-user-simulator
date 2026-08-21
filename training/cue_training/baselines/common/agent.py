"""The assistant side of the closed loop (LiteLLM)."""

from __future__ import annotations

from typing import Any

from cue_training.baselines.common.llm import achat, chat
from cue_training.baselines.common.prompts import agent_system_prompt

Turn = dict[str, str]


def _agent_messages(
    task: str, history: list[Turn], metadata: dict[str, Any]
) -> list[dict[str, str]]:
    """Build the assistant-side message list from the conversation so far.

    ``history`` ends with a user turn; its roles are passed through directly so
    the assistant model sees its own prior turns as ``assistant`` messages.
    """

    messages: list[dict[str, str]] = [
        {"role": "system", "content": agent_system_prompt(task, metadata)}
    ]
    for turn in history:
        role = turn.get("role", "user")
        messages.append(
            {
                "role": "assistant" if role == "assistant" else "user",
                "content": turn.get("content", ""),
            }
        )
    return messages


def agent_reply(
    task: str,
    history: list[Turn],
    metadata: dict[str, Any],
    *,
    model: str,
    api_key_env: str | None = "OPENAI_API_KEY",
    api_base: str | None = None,
    temperature: float = 0.7,
    max_tokens: int = 1024,
) -> str:
    """Generate the assistant's reply given the conversation so far."""

    return chat(
        _agent_messages(task, history, metadata),
        model=model,
        api_key_env=api_key_env,
        api_base=api_base,
        temperature=temperature,
        max_tokens=max_tokens,
    )


async def agent_reply_async(
    task: str,
    history: list[Turn],
    metadata: dict[str, Any],
    *,
    model: str,
    api_key_env: str | None = "OPENAI_API_KEY",
    api_base: str | None = None,
    temperature: float = 0.7,
    max_tokens: int = 1024,
) -> str:
    """Async counterpart of :func:`agent_reply` (uses ``achat``)."""

    return await achat(
        _agent_messages(task, history, metadata),
        model=model,
        api_key_env=api_key_env,
        api_base=api_base,
        temperature=temperature,
        max_tokens=max_tokens,
    )
