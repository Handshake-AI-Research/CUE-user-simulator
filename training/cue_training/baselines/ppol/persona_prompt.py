"""Shared PPol system-prompt helpers (train fitness + tau2 rollout).

Train (persona-policies ``injector.py``) and rollout (``PPolSimulator``) must build the
same system-prompt *structure*:

  1. tau2 native user-sim prompt (guidelines + ``<scenario>...``)
  2. CUE task-metadata block (``Task description:`` / ``Domain or topic:``)
  3. ``PERSONA_INJECTION_TEMPLATE`` wrapping the evolved persona (when present)

Keep the task-block labels byte-compatible with
``cue.decoder.mirror_prompts.user_proxy_task_block``.
"""

from __future__ import annotations

from typing import Any

PERSONA_INJECTION_TEMPLATE = """
## YOUR BEHAVIORAL STYLE FOR THIS CONVERSATION

In addition to your role and task above, you must embody the following behavioral persona throughout this conversation. This persona affects HOW you communicate, NOT WHAT you want or need. Your goal, preferences, and private information remain exactly as described above.

--- PERSONA POLICY BEGIN ---
{persona_policy_text}
--- PERSONA POLICY END ---

CRITICAL RULES FOR PERSONA ADHERENCE:
1. Your GOAL and PRIVATE INFORMATION do not change. You still want to accomplish the same task.
2. Apply the behavioral style consistently across ALL your turns, not just the first one.
3. If the persona says you are terse, be terse even when you have a lot to say.
4. If the persona says you withhold information, do not give it unless the agent specifically asks.
5. If the persona involves emotional state (impatience, confusion), let it show gradually and realistically.
6. Do NOT break character. Do NOT mention that you have a persona or behavioral instructions.
7. The persona should feel like a natural human interaction style, not a caricature.
""".strip()


def task_metadata_block(*, task_description: Any = None, domain: Any = None) -> str:
    """Same labels/format as ``user_proxy_task_block`` (no guidelines / trailer)."""

    lines: list[str] = []
    td = str(task_description).strip() if task_description else ""
    dm = str(domain).strip() if domain else ""
    if td:
        lines.append(f"Task description: {td}.")
    if dm:
        lines.append(f"Domain or topic: {dm}.")
    return "\n".join(lines)


def inject_persona_into_system_prompt(
    original_system_prompt: str,
    persona_policy_text: str,
    injection_template: str = PERSONA_INJECTION_TEMPLATE,
) -> str:
    """Append the persona block to the full user system prompt."""

    injected_block = injection_template.format(persona_policy_text=persona_policy_text.strip())
    return original_system_prompt.strip() + "\n\n" + injected_block.strip()


def append_task_metadata(system_prompt: str, *, task_description: Any = None, domain: Any = None) -> str:
    block = task_metadata_block(task_description=task_description, domain=domain)
    if not block:
        return system_prompt
    return f"{system_prompt.rstrip()}\n\n{block}"


def task_description_from_tau2_task(task: Any) -> str:
    """Best-effort short task text for the metadata block (reason_for_call when present)."""

    us = getattr(task, "user_scenario", None)
    if us is None and isinstance(task, dict):
        us = task.get("user_scenario")
    instr = getattr(us, "instructions", None) if us is not None else None
    if instr is None and isinstance(us, dict):
        instr = us.get("instructions")
    if instr is None:
        return ""
    reason = getattr(instr, "reason_for_call", None)
    if reason is None and isinstance(instr, dict):
        reason = instr.get("reason_for_call")
    if reason:
        return str(reason).strip()
    return str(instr).strip()
