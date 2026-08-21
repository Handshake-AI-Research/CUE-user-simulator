"""Native SimulatorArena user-simulator prompt templates (persona moved last)."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

from cue_training.evaluation.common.conditioning import Turn

_PROMPT_ROOT = (
    Path(__file__).resolve().parent
    / "external"
    / "SimulatorArena"
    / "simulation"
    / "prompts"
)


# The template's ``{user_profile}`` slot sits mid-prompt (writing) or first (math), so its
# style rules lose to the CoT/length instructions that follow. Render that slot empty and
# re-attach the persona last, where recency gives it a chance to win the conflict.
_PROFILE_FOOTER = """

# User Profile (highest priority)
{profile}

Follow this profile over any conflicting instruction above -- it governs how you write,
including message length, tone, capitalization, and punctuation."""

# The math templates open with "the characteristics defined in the profile below" and give
# ``{user_profile}`` a section of its own, so blanking the slot would leave a dangling empty
# header contradicting that sentence. Point at the real block instead. (The writing templates
# keep their objectives and pre-writing materials in that section, so it stays populated.)
_PROFILE_POINTER = 'See "User Profile (highest priority)" at the end of this prompt.'

# The tutor answers in LaTeX in ~99% of its turns and the simulator mirrors it: 33.6% of CUE's
# math user turns carried LaTeX against 0.3% for real students, and 526 of those 527 turns came
# straight after a LaTeX tutor reply (only 6.3% of opening turns, before any tutor reply, had
# any). Real students type math the way it goes into a chat box, so say so for every math arm.
_MATH_PLAINTEXT = """

# How to type math
Type math as plain text, the way you would in a chat box: "r^2 + 5r + 7 = 0", "49 - 36 = 13".
Do not use LaTeX, dollar signs, or \\( \\) \\[ \\] delimiters, and do not copy the tutor's
formatting -- write the math the way you would type it yourself."""


def _with_profile(rendered: str, user_profile: str) -> str:
    profile = (user_profile or "").strip()
    if not profile:
        return rendered
    return rendered.rstrip() + _PROFILE_FOOTER.format(profile=profile)


def _conversation_history(history: list[Turn], domain: str) -> str:
    assistant = "AI Tutor" if domain == "math" else "AI Writing Assistant"
    lines: list[str] = []
    for turn in history:
        role = "You" if turn.get("role") == "user" else assistant
        lines.append(f"- {role}: {turn.get('content', '')}")
    return "\n".join(lines)


def native_user_prompt(
    *,
    domain: str,
    history: list[Turn],
    user_profile: str,
    task: str,
    document_type: Any = None,
    intent: Any = None,
    background: Any = None,
    math_problem: Any = None,
) -> str:
    """Render SimulatorArena's profile-conditioned prompt, persona appended last."""

    prompt_dir = "math_tutoring" if domain == "math" else "document_creation"
    suffix = "-initial-query" if not history else ""
    template = (_PROMPT_ROOT / prompt_dir / f"zero-shot-cot-user-profile{suffix}.txt").read_text(
        encoding="utf-8"
    )
    slot = _PROFILE_POINTER if (user_profile or "").strip() else ""
    if domain == "math":
        rendered = template.format(
            user_profile=slot,
            math_problem=math_problem or task,
            conversation_history=_conversation_history(history, domain),
            focus_feature_text="",
        ) + _MATH_PLAINTEXT
    else:
        rendered = template.format(
            user_profile=slot,
            document_type=document_type or "document",
            intent=intent or task,
            pre_writing_materials=background or "",
            conversation_history=_conversation_history(history, domain),
        )
    return _with_profile(rendered, user_profile)


# Models routinely answer with the template's own section headings (``## Thought Process`` /
# ``## Message``) instead of the ``Thought:`` / ``Message:`` labels the output format asks for,
# and echo the ``**initial message**`` bold as a leading ``**``. Matching only "Thought:" let
# the whole scratchpad through as the user's turn, so accept the heading forms too.
# The label must carry a colon or stand alone on its line, else a student turn that opens
# "Response to the hint made sense..." would be cut down to "to the hint made sense...".
_MESSAGE_MARKER = re.compile(
    r"(?:^|\n)[ \t]*(?:#{1,4}[ \t]*)?\**(?:Message|Response|Query)"
    r"(?:[ \t]+Generation)?\**[ \t]*(?::[ \t]*|(?=\n|$))"
)


def parse_native_user_output(text: str) -> tuple[str, bool]:
    """Apply SimulatorArena's Thought/Message parsing and termination contract."""

    raw = (text or "").strip()
    if not raw or "terminate conversation" in raw.lower():
        return "", True
    marker = _MESSAGE_MARKER.search(raw)
    if marker:
        raw = raw[marker.end():].strip()
    # Only bold/heading runs, so a message that really opens with "* item" keeps its bullet.
    raw = re.sub(r"^\s*(?:\*\*+|#+)\s*", "", raw)
    return raw, not raw
