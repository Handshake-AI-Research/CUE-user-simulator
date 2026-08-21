"""Versioned persona-manual and training-record schemas."""

from __future__ import annotations

import json
import re
from typing import Any

SCHEMA_VERSION = 1
COMMAND_BLOCK_SENTINEL = "<COMMAND_BLOCK>\n"
GENERAL_COMMAND_SENTINEL = "<GENERAL_COMMAND>\n"
USER_SPECIFIC_SENTINEL = "<USER_SPECIFIC>\n"
STYLE_COMMAND_SENTINEL = "<STYLE_COMMAND>\n"
COMMAND_SLOT_SENTINEL = "<COMMAND_SLOT>\n"
MAX_COMMANDS = 15
MAX_EXAMPLES_PER_COMMAND = 4
MAX_TOTAL_EXAMPLES = 4
_MAX_COMMAND_CHARS = 240
_MAX_EXAMPLE_CHARS = 200
_KINDS = frozenset({"sim_contrast", "human_contrast", "style"})
_FORMS = frozenset({"general", "if_then"})
_JSON_RE = re.compile(r"\{.*\}", re.DOTALL)
_ASK_DIRECTIVE_RE = re.compile(
    r"(?:^|[.;]\s*|\b(?:instead|then|to)\s+)ask\b"
    r"|\b(?:you|user|person|traveler|customer|they|who)\s+asks?\s+for\b"
    r"|\bask\s+(?:the\s+(?:assistant|agent)\s+)?"
    r"(?:for|about|whether|what|which|when|where|who|why|how)\b",
    re.IGNORECASE,
)
_META_EXAMPLE_RE = re.compile(
    r"^(?:if|when)\b|^(?:the\s+)?(?:assistant|agent)\b|(?:->|=>)",
    re.IGNORECASE,
)
_QUOTED_REPLY_RE = re.compile(
    r"\b(?:reply|say|answer|respond)(?:\s+with)?\s+[\"“]([^\"”]+)[\"”]",
    re.IGNORECASE,
)


def clean_generated_command(text: str) -> str:
    """Normalize one command-slot continuation into a single command string."""

    cleaned = str(text or "").strip()
    if not cleaned:
        return ""
    first = next((line.strip() for line in cleaned.splitlines() if line.strip()), "")
    first = re.sub(r"^(?:[-*•]|\d+[.)])\s*", "", first)
    first = first.strip("`\"' \t")
    if first.casefold() in {"none", "no command", "<no_command>"}:
        return ""
    return first[:_MAX_COMMAND_CHARS]


class ManualValidationError(ValueError):
    pass


def _coerce_str_list(value: Any, *, limit: int, max_chars: int) -> list[str]:
    if isinstance(value, str):
        value = [value]
    if not isinstance(value, (list, tuple)):
        return []
    out: list[str] = []
    for item in value:
        text = str(item or "").strip()
        if not text:
            continue
        out.append(text[:max_chars])
        if len(out) >= limit:
            break
    return out


def _unique_examples(values: list[str]) -> list[str]:
    out: list[str] = []
    seen: set[str] = set()
    for value in values:
        key = value.casefold()
        if key in seen:
            continue
        seen.add(key)
        out.append(value)
        if len(out) >= MAX_TOTAL_EXAMPLES:
            break
    return out


def validate_manual(block: Any) -> dict[str, Any]:
    """Coerce into the canonical schema or raise."""

    if not isinstance(block, dict):
        raise ManualValidationError("manual must be an object")
    raw = block.get("commands")
    if not isinstance(raw, (list, tuple)) or not raw:
        raise ManualValidationError("commands must be a nonempty list")
    commands: list[dict[str, Any]] = []
    seen: set[str] = set()
    for index, entry in enumerate(raw):
        if isinstance(entry, str):
            entry = {"text": entry}
        if not isinstance(entry, dict):
            continue
        text = str(entry.get("text") or entry.get("command") or "").strip()
        if not text:
            continue
        key = text.lower()
        if key in seen:
            continue
        seen.add(key)
        kind = str(entry.get("kind") or ("sim_contrast" if index < 5 else "human_contrast"))
        form = str(entry.get("form") or "general")
        if kind not in _KINDS:
            kind = "human_contrast"
        if form not in _FORMS:
            form = "general"
        commands.append(
            {
                "text": text[:_MAX_COMMAND_CHARS],
                "kind": kind,
                "form": form,
                "examples": _coerce_str_list(
                    entry.get("examples"),
                    limit=MAX_EXAMPLES_PER_COMMAND,
                    max_chars=_MAX_EXAMPLE_CHARS,
                ),
                "evidence_turn_ids": [
                    str(x) for x in (entry.get("evidence_turn_ids") or []) if str(x).strip()
                ],
            }
        )
        if len(commands) >= MAX_COMMANDS:
            break
    if not commands:
        raise ManualValidationError("no usable commands")
    return {"schema_version": SCHEMA_VERSION, "commands": commands}


def canonicalize_manual(block: dict[str, Any]) -> str:
    validated = validate_manual(block)
    return json.dumps(validated, ensure_ascii=False, sort_keys=True)


def to_profile(block: Any) -> dict[str, Any]:
    """Convert a cue manual (or already-flat profile) to train/decode shape.

    ::
        {"commands": [str, ...], "examples": [str, ...]}
    """

    if not isinstance(block, dict):
        raise ManualValidationError("manual must be an object")
    commands: list[str] = []
    examples: list[str] = []
    raw = block.get("commands")
    if isinstance(raw, (list, tuple)) and raw and all(isinstance(c, str) for c in raw):
        seen_commands: set[str] = set()
        for item in raw:
            text = str(item or "").strip()
            key = text.casefold()
            if text and key not in seen_commands:
                commands.append(text[:_MAX_COMMAND_CHARS])
                seen_commands.add(key)
        examples = _coerce_str_list(
            block.get("examples"),
            limit=MAX_COMMANDS * MAX_EXAMPLES_PER_COMMAND,
            max_chars=_MAX_EXAMPLE_CHARS,
        )
    else:
        validated = validate_manual(block)
        for cmd in validated["commands"]:
            text = str(cmd.get("text") or "").strip()
            if text:
                commands.append(text[:_MAX_COMMAND_CHARS])
            for ex in cmd.get("examples") or []:
                ex_s = str(ex).strip()
                if ex_s:
                    examples.append(ex_s[:_MAX_EXAMPLE_CHARS])
    if not commands:
        raise ManualValidationError("no usable commands")
    return {
        "commands": commands,
        "examples": _unique_examples(examples),
    }


def canonicalize_profile(block: dict[str, Any]) -> str:
    """Train/decode target: {commands, examples} only (no evidence ids / kinds)."""

    return json.dumps(to_profile(block), ensure_ascii=False, sort_keys=True)


def filter_joint_training_target(persona_manual: dict[str, Any]) -> dict[str, Any]:
    """Remove assistant-like directives and procedural examples from joint-training targets."""

    validated = validate_manual(persona_manual)
    commands: list[dict[str, Any]] = []
    for command in validated["commands"]:
        text = str(command.get("text") or "").strip()
        if _ASK_DIRECTIVE_RE.search(text):
            continue
        examples: list[str] = []
        for raw_example in command.get("examples") or []:
            example = str(raw_example).strip()
            quoted = _QUOTED_REPLY_RE.search(example)
            if quoted:
                example = quoted.group(1).strip()
            elif _META_EXAMPLE_RE.search(example):
                continue
            if example and len(example.split()) <= 12:
                examples.append(example)
        commands.append({**command, "examples": examples})
    if not commands:
        commands = [
            {
                "text": "Answer as the user in short, natural language.",
                "kind": "sim_contrast",
                "form": "general",
                "examples": [],
                "evidence_turn_ids": [],
            }
        ]
    return {"schema_version": SCHEMA_VERSION, "commands": commands}


def split_manual_targets(
    persona_manual: dict[str, Any],
) -> tuple[str, str, str]:
    """Split a data-annotation manual into general / user-specific / style decode targets.

    - ``sim_contrast`` -> general anti-assistant behavior
    - ``human_contrast`` (+ examples) -> user-specific residual
    - ``style`` -> surface writing habits (casing, fragments, punctuation, etc.)
    """

    validated = validate_manual(persona_manual)
    general_cmds: list[str] = []
    specific_cmds: list[str] = []
    style_cmds: list[str] = []
    specific_examples: list[str] = []
    for cmd in validated["commands"]:
        text = str(cmd.get("text") or "").strip()
        if not text:
            continue
        kind = str(cmd.get("kind") or "human_contrast")
        examples = [str(x).strip() for x in (cmd.get("examples") or []) if str(x).strip()]
        if kind == "sim_contrast":
            general_cmds.append(text)
        elif kind == "style":
            style_cmds.append(text)
        else:
            specific_cmds.append(text)
            specific_examples.extend(examples)
    if not general_cmds and specific_cmds:
        split_at = max(1, len(specific_cmds) // 2)
        general_cmds = specific_cmds[:split_at]
        specific_cmds = specific_cmds[split_at:]
    if not specific_cmds and general_cmds:
        split_at = max(1, len(general_cmds) // 2)
        specific_cmds = general_cmds[split_at:]
        general_cmds = general_cmds[:split_at]
    if not general_cmds:
        general_cmds = ["Speak naturally as the user, not the assistant."]
    if not specific_cmds:
        specific_cmds = (
            [general_cmds.pop()]
            if len(general_cmds) > 1
            else ["Follow the user's conversational style."]
        )
    general = canonicalize_profile({"commands": general_cmds, "examples": []})
    specific = canonicalize_profile(
        {"commands": specific_cmds, "examples": _unique_examples(specific_examples)}
    )
    if style_cmds:
        style = canonicalize_profile({"commands": style_cmds, "examples": []})
    else:
        style = json.dumps({"commands": [], "examples": []}, ensure_ascii=False, sort_keys=True)
    return general, specific, style


def merge_dual_manual_blocks(
    general: dict[str, Any] | None,
    specific: dict[str, Any] | None,
    style: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Merge parsed general + user-specific (+ optional style) into one manual."""

    g = general or {"commands": [], "examples": []}
    s = specific or {"commands": [], "examples": []}
    st = style or {"commands": [], "examples": []}
    return {
        "commands": (
            list(g.get("commands") or [])
            + list(s.get("commands") or [])
            + list(st.get("commands") or [])
        ),
        "examples": _unique_examples(
            list(g.get("examples") or [])
            + list(s.get("examples") or [])
            + list(st.get("examples") or [])
        ),
    }


def render_manual(block: dict[str, Any] | None) -> str:
    if block is None:
        return ""
    try:
        return canonicalize_manual(block)
    except ManualValidationError:
        return ""


def parse_manual(text: str) -> dict[str, Any] | None:
    if not text:
        return None
    cleaned = text.strip()
    for sentinel in (
        COMMAND_BLOCK_SENTINEL.strip(),
        GENERAL_COMMAND_SENTINEL.strip(),
        USER_SPECIFIC_SENTINEL.strip(),
        STYLE_COMMAND_SENTINEL.strip(),
    ):
        if sentinel in cleaned:
            cleaned = cleaned.split(sentinel, 1)[1].strip()
    if cleaned.startswith("```"):
        cleaned = cleaned.strip("`")
        cleaned = re.sub(r"^(json|JSON)\s*", "", cleaned).strip()
    match = _JSON_RE.search(cleaned)
    for candidate in (cleaned, match.group(0) if match else None):
        if not candidate:
            continue
        try:
            data = json.loads(candidate)
        except (json.JSONDecodeError, TypeError):
            continue
        if not isinstance(data, dict):
            continue
        # Prefer flat profile shape (string commands) — used as train/decode targets.
        raw_cmds = data.get("commands")
        if isinstance(raw_cmds, list) and all(isinstance(c, str) for c in raw_cmds):
            if not raw_cmds:
                return {"commands": [], "examples": []}
            try:
                return to_profile(data)
            except ManualValidationError:
                continue
        try:
            return validate_manual(data)
        except (TypeError, ManualValidationError):
            continue
    return None


def validate_record(record: Any) -> dict[str, Any]:
    if not isinstance(record, dict):
        raise ValueError("record must be an object")
    session_id = str(record.get("id") or record.get("session_id") or "").strip()
    if not session_id:
        raise ValueError("record requires id/session_id")
    turns = record.get("turns")
    if not isinstance(turns, list) or not turns:
        raise ValueError("record requires nonempty turns")
    normalized_turns: list[dict[str, Any]] = []
    for index, turn in enumerate(turns):
        if not isinstance(turn, dict):
            continue
        role = str(turn.get("role") or "").strip().lower()
        content = str(turn.get("content") or turn.get("user_text") or turn.get("system_text") or "").strip()
        if role not in {"user", "assistant", "system"}:
            # Accept CUE-style flat turns with user_text/system_text.
            user_text = str(turn.get("user_text") or "").strip()
            system_text = str(turn.get("system_text") or "").strip()
            if system_text:
                normalized_turns.append(
                    {
                        "turn_id": f"{session_id}:sys:{index}",
                        "role": "assistant",
                        "content": system_text,
                    }
                )
            if user_text:
                normalized_turns.append(
                    {
                        "turn_id": str(turn.get("turn_id") or f"{session_id}:user:{index}"),
                        "role": "user",
                        "content": user_text,
                    }
                )
            continue
        if not content:
            continue
        normalized_turns.append(
            {
                "turn_id": str(turn.get("turn_id") or f"{session_id}:{role}:{index}"),
                "role": role,
                "content": content,
            }
        )
    if not any(t["role"] == "user" for t in normalized_turns):
        raise ValueError("record requires at least one nonempty user turn")
    manual = record.get("persona_manual")
    if manual is not None:
        manual = validate_manual(manual)
    return {
        "id": session_id,
        "session_id": session_id,
        "dataset": str(record.get("dataset") or "unknown"),
        "split": str(record.get("split") or "train"),
        "turns": normalized_turns,
        "persona_manual": manual,
        "provenance": record.get("provenance") or {},
        "metadata": record.get("metadata") or {},
    }
