"""Render manuals as steering text for a user simulator prompt."""

from __future__ import annotations

from typing import Any

# Surface form is the point of these examples: register (punctuation, casing, length) is
# what a sim2real discriminator reads, and the model's own polished default overrides an
# abstract rule, so the mechanics have to be named explicitly. But these are annotator
# sketches, not transcribed messages, so point at the mechanics rather than the wording
# and say what the placeholders are, or the model copies prose verbatim (which reads
# synthetic) or emits the brackets.
_EXAMPLES_HEADER = (
    "Style samples -- short sketches of how users like this one type. Match their "
    "mechanics, not their wording: punctuation (including missing, doubled, or absent "
    "end marks), capitalization, spacing, contractions, and how short the messages are. "
    "Angle-bracket slots like <ENTITY> or <NUM> stand in for details -- put your own "
    "specifics there and never write the brackets. Take nothing else from them, "
    "including their task content:\n"
)


def render_manual(manual: dict[str, Any] | None) -> str:
    """Render only simulator-safe commands and examples."""

    if not manual:
        return ""
    commands: list[str] = []
    examples: list[str] = []
    for entry in manual.get("commands") or []:
        if isinstance(entry, dict):
            text = str(entry.get("text") or entry.get("command") or "").strip()
            examples.extend(str(x).strip() for x in entry.get("examples") or [] if str(x).strip())
        else:
            text = str(entry).strip()
        if text:
            commands.append(text)
    examples.extend(str(x).strip() for x in manual.get("examples") or [] if str(x).strip())
    sections = []
    if commands:
        sections.append("Behavior manual:\n" + "\n".join(f"- {text}" for text in commands))
    if examples:
        sections.append(_EXAMPLES_HEADER + "\n".join(f"- {text}" for text in examples))
    return "\n\n".join(sections)


def render_dual_manual(
    general: dict[str, Any] | None,
    specific: dict[str, Any] | None,
    style: dict[str, Any] | None = None,
) -> str:
    """Render general + user-specific (+ optional style) sections for steering."""

    sections: list[str] = []
    gcmds = [str(c).strip() for c in (general or {}).get("commands") or [] if str(c).strip()]
    scmds = [str(c).strip() for c in (specific or {}).get("commands") or [] if str(c).strip()]
    stcmds = [str(c).strip() for c in (style or {}).get("commands") or [] if str(c).strip()]
    examples = [
        str(x).strip()
        for x in (list((specific or {}).get("examples") or []) + list((style or {}).get("examples") or []))
        if str(x).strip()
    ]
    if gcmds:
        sections.append("General behavior:\n" + "\n".join(f"- {text}" for text in gcmds))
    if scmds:
        sections.append("User-specific behavior:\n" + "\n".join(f"- {text}" for text in scmds))
    if stcmds:
        sections.append("Writing style:\n" + "\n".join(f"- {text}" for text in stcmds))
    if examples:
        sections.append(_EXAMPLES_HEADER + "\n".join(f"- {text}" for text in examples))
    return "\n\n".join(sections)
