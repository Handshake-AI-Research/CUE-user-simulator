"""Prompt builders shared by the agent and prompt-based user simulators."""

from __future__ import annotations

from typing import Any

Turn = dict[str, str]

_DOMAIN_AGENT_HINT = {
    "coding": "You are a coding assistant helping the user write, fix, and reason about code.",
    "customer_service": "You are a customer-service agent helping the user resolve their request.",
    "writing": "You are a writing assistant helping the user draft and refine documents.",
}


def transcript_text(history: list[Turn], max_turns: int = 16) -> str:
    if not history:
        return "(no messages yet)"
    lines = []
    for turn in history[-max_turns:]:
        role = "User" if turn.get("role") == "user" else "Assistant"
        lines.append(f"{role}: {turn.get('content', '')}")
    return "\n".join(lines)


def agent_system_prompt(task: str, metadata: dict[str, Any]) -> str:
    domain = str(metadata.get("domain", ""))
    hint = _DOMAIN_AGENT_HINT.get(domain, "You are a helpful AI assistant.")
    return (
        f"{hint} Be concise and directly address the user's request. "
        "Respond only as the assistant."
    )


def user_system_prompt(
    task: str,
    metadata: dict[str, Any],
    augmentation: str = "",
) -> str:
    base = (
        "You are role-playing as the HUMAN USER in a task-oriented conversation with an AI "
        "assistant. Stay in the user role and pursue your goal across turns. Write like a real "
        "person: you may be brief, informal, or imprecise, and you do not need to be polished. "
        "Do not act as the assistant or solve the task yourself. Output ONLY the next user "
        "message, with no quotes, role labels, or meta-commentary."
    )
    if augmentation:
        base = base + "\n\n" + augmentation.strip()
    return base


def user_turn_prompt(task: str, history: list[Turn]) -> str:
    return (
        f"Your goal / intent:\n{task}\n\n"
        f"Conversation so far:\n{transcript_text(history)}\n\n"
        "Write ONLY your next user message:"
    )
