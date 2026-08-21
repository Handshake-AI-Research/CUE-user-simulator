"""MirrorBench-compatible user-proxy prompt blocks."""

import re
from typing import Any

# Reasoning / canvas / tool wrappers some chat models emit around their actual reply.
_THINK_BLOCK = re.compile(r"<\|?\s*think(?:ing)?\s*\|?>.*?<\|?\s*/\s*think(?:ing)?\s*\|?>", re.DOTALL | re.IGNORECASE)
_HTML_THINK = re.compile(r"<think>.*?</think>", re.DOTALL | re.IGNORECASE)
_CONTROL_TOKEN = re.compile(r"<\|?\s*/?\s*(?:think(?:ing)?|canvas|tool|reasoning|assistant|system|user)\s*\|?>", re.IGNORECASE)


def trim_control_message(text: Any, *, return_stop: bool = False) -> str | tuple[str, bool]:
    """Strip reasoning/control wrappers, role prefixes, quotes, and cut at ``###``.

    When ``return_stop`` is False (default), returns only the cleaned utterance (empty
    when the model produced only control/stop tokens). When True, returns
    ``(utterance, stopped)`` so callers can end the episode on a mid-turn ``###``
    (same contract as ``clean_user_message``).
    """

    from cue_training.evaluation.common.rollout import clean_user_message

    s = str(text or "")
    s = _THINK_BLOCK.sub("", s)
    s = _HTML_THINK.sub("", s)
    s = _CONTROL_TOKEN.sub("", s)
    return clean_user_message(s, return_stop=return_stop)


def _normalise_text(value: Any) -> str | None:
    if not value:
        return None
    text = str(value).strip()
    return text or None


def user_proxy_system_prompt(
    *,
    task_description: Any = None,
    domain: Any = None,
    persona: Any = None,
) -> str:
    """Replicate ``build_user_proxy_system_prompt`` from MirrorBench."""

    task_description = _normalise_text(task_description)
    domain = _normalise_text(domain)
    persona = _normalise_text(persona)
    prompt_lines = [
        "You are simulating a real human user for the MirrorBench evaluation harness.",
        "Respond with the next USER turn only. Do not write assistant messages, notes, or any other analysis.",
        "Speak ONLY as the user who wants the task done. Never act as the assistant/agent: do not "
        "solve the task yourself, write code, run commands, or describe the steps you are taking.",
        "Do NOT narrate your reasoning or plan. Never think out loud or open with phrases like "
        "\"Okay, I need to...\", \"Let me...\", or \"First, I'll...\". Just say what the user would type.",
        "Never write or paste code, files, diffs, or command output -- ask the assistant for them "
        "in plain prose. Never begin with a role name or chat header such as \"assistant\" or "
        "\"User:\".",
        "Your utterance should be like a real user and the context should be based on the following information provided.",
    ]
    if task_description:
        prompt_lines.append(f"Task description: {task_description}.")
    if domain:
        prompt_lines.append(f"Domain or topic: {domain}.")
    if persona:
        prompt_lines.append(f"Persona hints: {persona}.")
    if not any([task_description, domain, persona]):
        prompt_lines.append(
            "No additional dataset metadata provided. Respond naturally and plausibly based on the ongoing conversation."
        )
    prompt_lines.append(
        "Match the length, tone, and specificity of real user utterances. If you are unsure, "
        "respond naturally based on the assistant's previous messages like how a real human would. "
        "Note that your response MUST not contain anything other than the USER utterance. Do not "
        "include any prefixes like 'User:' or 'Human:' as well. Just the raw message content."
    )
    return "\n".join(prompt_lines)


def user_proxy_task_block(
    *,
    task_description: Any = None,
    domain: Any = None,
    persona: Any = None,
) -> str:
    """Return only the dataset-metadata lines from the full proxy prompt."""

    lines: list[str] = []
    task = _normalise_text(task_description)
    topic = _normalise_text(domain)
    hints = _normalise_text(persona)
    if task:
        lines.append(f"Task description: {task}.")
    if topic:
        lines.append(f"Domain or topic: {topic}.")
    if hints:
        lines.append(f"Persona hints: {hints}.")
    return "\n".join(lines)
