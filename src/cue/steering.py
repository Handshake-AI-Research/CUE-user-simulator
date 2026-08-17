"""Build the steering text a user simulator receives."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from collections.abc import Sequence

# Same guidance the research stack uses: match typing mechanics, not task content.
_EXAMPLES_HEADER = (
    "Style samples -- short sketches of how users like this one type. Match their "
    "mechanics, not their wording: punctuation (including missing, doubled, or absent "
    "end marks), capitalization, spacing, contractions, and how short the messages are. "
    "Angle-bracket slots like <ENTITY> or <NUM> stand in for details -- put your own "
    "specifics there and never write the brackets. Take nothing else from them, "
    "including their task content:\n"
)


def _as_commands(items: Sequence[Any] | None) -> list[str]:
    out: list[str] = []
    for entry in items or []:
        if isinstance(entry, dict):
            text = str(entry.get("text") or entry.get("command") or "").strip()
        else:
            text = str(entry).strip()
        if text:
            out.append(text)
    return out


def render_steering(
    *,
    general: Sequence[Any] | None = None,
    specific: Sequence[Any] | None = None,
    style: Sequence[Any] | None = None,
    commands: Sequence[Any] | None = None,
    examples: Sequence[Any] | None = None,
) -> str:
    """Sectioned steering prompt (general / user-specific / style / samples)."""

    g = _as_commands(general)
    s = _as_commands(specific)
    st = _as_commands(style)
    flat = _as_commands(commands)
    ex = [str(x).strip() for x in (examples or []) if str(x).strip()]

    sections: list[str] = []
    if g or s or st:
        if g:
            sections.append("General behavior:\n" + "\n".join(f"- {c}" for c in g))
        if s:
            sections.append("User-specific behavior:\n" + "\n".join(f"- {c}" for c in s))
        if st:
            sections.append("Writing style:\n" + "\n".join(f"- {c}" for c in st))
    elif flat:
        sections.append("Behavior manual:\n" + "\n".join(f"- {c}" for c in flat))
    if ex:
        sections.append(_EXAMPLES_HEADER + "\n".join(f"- {text}" for text in ex))
    return "\n\n".join(sections)


def user_sim_system(
    scenario: str,
    manual_text: str | None = None,
    *,
    base: str | None = None,
) -> str:
    """System prompt for a role-played user, optionally steered by a CUE manual."""

    body = (
        base
        or (
            "You are role-playing a human user talking to an AI assistant. Stay in character, "
            "send exactly one short message per turn, and never speak or act as the assistant.\n\n"
            "Your goal in this conversation:\n{scenario}"
        )
    ).format(scenario=scenario.strip())
    addon = (manual_text or "").strip()
    if not addon:
        return body
    return f"{body}\n\nBehave like this specific user:\n{addon}"
