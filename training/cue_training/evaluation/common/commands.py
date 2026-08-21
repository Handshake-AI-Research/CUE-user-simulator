"""Validation and rendering for baseline command blocks."""

import os
from typing import Any

MAX_COMMANDS = int(os.environ.get("DECODER_MAX_COMMANDS", "10"))
MAX_EXAMPLES_PER_COMMAND = 4
MAX_TOTAL_EXAMPLES = 4
_MAX_COMMAND_CHARS = 240
_MAX_EXAMPLE_CHARS = 200


def _coerce_str_list(value: Any, *, limit: int, max_chars: int) -> list[str]:
    if isinstance(value, str):
        value = [value]
    if not isinstance(value, (list, tuple)):
        return []
    out = []
    for item in value:
        text = str(item or "").strip()
        if text:
            out.append(text[:max_chars])
        if len(out) >= limit:
            break
    return out


def validate_command_block(block: Any) -> dict[str, Any] | None:
    if not isinstance(block, dict) or not isinstance(block.get("commands"), (list, tuple)):
        return None
    commands = []
    seen = set()
    for entry in block["commands"]:
        if isinstance(entry, str):
            entry = {"command": entry, "examples": []}
        if not isinstance(entry, dict):
            continue
        command = str(entry.get("command") or entry.get("text") or "").strip()
        if not command or command.lower() in seen:
            continue
        seen.add(command.lower())
        commands.append(
            {
                "command": command[:_MAX_COMMAND_CHARS],
                "examples": _coerce_str_list(
                    entry.get("examples"),
                    limit=MAX_EXAMPLES_PER_COMMAND,
                    max_chars=_MAX_EXAMPLE_CHARS,
                ),
            }
        )
        if len(commands) >= MAX_COMMANDS:
            break
    return {"commands": commands} if commands else None


def render_command_block(block: dict[str, Any] | None) -> str:
    block = validate_command_block(block) if block is not None else None
    if not block:
        return ""
    lines = ["You are imitating ONE specific real user.", "Behavioral rules (follow these):"]
    examples = []
    for entry in block["commands"]:
        lines.append(f"- {entry['command']}")
        examples.extend(entry["examples"])
    if examples:
        lines.append("Example phrasings (match this style):")
        unique_examples = list(dict.fromkeys(examples))[:MAX_TOTAL_EXAMPLES]
        lines.extend(f'- "{example}"' for example in unique_examples)
    lines.append(
        "Real users are often terse, make typos, and skip pleasantries. Do not sound "
        "like a helpful AI; sound like this specific person."
    )
    return "\n".join(lines)
