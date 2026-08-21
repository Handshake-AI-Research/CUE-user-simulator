"""Paper Appendix C.2 persona-block formatting for RealUserSim."""

from __future__ import annotations

from typing import Any

DEMOGRAPHIC_LABELS = {
    "age": "Age",
    "gender": "Gender",
    "education": "Education",
    "occupation": "Occupation",
    "location": "Location",
    "income": "Income",
    "marital_status": "Marital status",
    "nationality": "Nationality",
}

DEMOGRAPHIC_FIELDS = tuple(DEMOGRAPHIC_LABELS.keys())

# Exact paper anti-normalization constraint (RealUserSim §4.1 / PT3 setup).
ANTI_NORMALIZATION = (
    'Do not "clean up" the writing. If the commands require poor grammar and '
    "frequent typos, your response must be equally messy."
)


def _demo_value(info: Any) -> tuple[str | None, str | None]:
    if isinstance(info, dict):
        value = info.get("value")
        source = info.get("source")
        if value is None or str(value).lower() in ("null", "none", "unknown", ""):
            return None, None
        return str(value).strip(), (str(source).strip() if source else None)
    if info is None or str(info).lower() in ("null", "none", "unknown", ""):
        return None, None
    return str(info).strip(), None


def command_example_pairs(profile: dict[str, Any]) -> list[tuple[str, list[str]]]:
    """Normalize to (command, examples) pairs from paper or legacy schemas."""

    pairs: list[tuple[str, list[str]]] = []
    raw_pairs = profile.get("command_examples")
    if isinstance(raw_pairs, list) and raw_pairs:
        for item in raw_pairs:
            if isinstance(item, dict):
                cmd = str(item.get("command") or "").strip()
                ex = item.get("examples") or []
                if isinstance(ex, str):
                    ex = [ex]
                examples = [str(e).strip() for e in ex if str(e).strip()]
                if cmd:
                    pairs.append((cmd, examples))
            elif isinstance(item, str) and item.strip():
                pairs.append((item.strip(), []))
        return pairs[:15]

    # Legacy flat lists: zip when lengths match; otherwise attach all examples to first command.
    commands = [str(c).strip() for c in (profile.get("commands") or []) if str(c).strip()]
    examples = [str(e).strip() for e in (profile.get("examples") or []) if str(e).strip()]
    if not commands:
        return []
    if len(commands) == len(examples) and examples:
        return [(c, [e]) for c, e in zip(commands, examples)][:15]
    if len(commands) == 1:
        return [(commands[0], examples[:4])]
    out: list[tuple[str, list[str]]] = [(c, []) for c in commands]
    if examples and out:
        out[0] = (out[0][0], examples[:4])
    return out[:15]


def strip_profile_examples(profile: dict[str, Any]) -> dict[str, Any]:
    """Drop verbatim examples (leakage control for paired eval extraction)."""

    out = dict(profile)
    pairs = command_example_pairs(out)
    out["command_examples"] = [{"command": cmd, "examples": []} for cmd, _ex in pairs]
    out["commands"] = [cmd for cmd, _ex in pairs]
    out["examples"] = []
    out["include_examples"] = False
    return out


def format_persona_block(
    profile: dict[str, Any] | str,
    *,
    include_examples: bool | None = None,
) -> str:
    """Assemble the paper ``{persona block}`` (demographics + background + Command+Example)."""

    if isinstance(profile, str):
        return profile.strip()

    if include_examples is None:
        include_examples = profile.get("include_examples", True) is not False

    # Prefer an already-rendered sectioned manual when present (CUE Stage-0 / bakeoff path).
    # Salesforce/RealUserSim published profiles use ``linguistic_profile`` for the same block.
    # Paired-noex must not emit verbatim Examples from a pre-rendered manual.
    manual_sections = str(
        profile.get("manual_sections") or profile.get("linguistic_profile") or ""
    ).strip()
    if manual_sections and include_examples:
        # Published profiles often store only the Command/Examples body; still prepend demos.
        demo = profile.get("demographics") if isinstance(profile.get("demographics"), dict) else {}
        if demo and not manual_sections.lower().startswith("demographic"):
            demo_lines: list[str] = []
            for field, label in DEMOGRAPHIC_LABELS.items():
                value, source = _demo_value(demo.get(field))
                if not value:
                    continue
                suffix = f" (source: {source})" if source in ("extracted", "inferred") else ""
                demo_lines.append(f"- {label}: {value}{suffix}")
            if demo_lines:
                return "Demographics:\n" + "\n".join(demo_lines) + "\n\n" + manual_sections
        return manual_sections

    sections: list[str] = []
    demo = profile.get("demographics") if isinstance(profile.get("demographics"), dict) else {}
    demo_lines: list[str] = []
    for field, label in DEMOGRAPHIC_LABELS.items():
        value, source = _demo_value(demo.get(field))
        if not value:
            continue
        suffix = f" (source: {source})" if source in ("extracted", "inferred") else ""
        demo_lines.append(f"- {label}: {value}{suffix}")
    # Preserve any extra lean keys (expertise/tone/...) if rich fields are empty.
    if not demo_lines:
        for k, v in demo.items():
            value, source = _demo_value(v)
            if value:
                suffix = f" (source: {source})" if source in ("extracted", "inferred") else ""
                demo_lines.append(f"- {k.replace('_', ' ').title()}: {value}{suffix}")
    if demo_lines:
        sections.append("Demographics:\n" + "\n".join(demo_lines))

    background = str(profile.get("additional_background") or "").strip()
    if background:
        sections.append(f"Additional background:\n{background}")

    style_lines: list[str] = []
    for command, examples in command_example_pairs(profile):
        style_lines.append(f"Command: {command}")
        if include_examples and examples:
            quoted = "; ".join(f'"{e}"' for e in examples)
            style_lines.append(f"Examples: {quoted}")
    if style_lines:
        sections.append("Communication Style Instructions:\n" + "\n".join(style_lines))
    return "\n\n".join(sections).strip()
