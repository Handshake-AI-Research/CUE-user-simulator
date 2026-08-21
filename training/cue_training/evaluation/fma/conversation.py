"""Normalize tau2 / τ-USI conversations into a uniform tool-aware turn list."""

from __future__ import annotations

import json
import re
from typing import Any

Turn = dict[str, Any]

_CANVAS_RE = re.compile(r"<\|canvas\|>.*?(?:<\|/canvas\|>|$)", re.DOTALL)
_THINK_RE = re.compile(r"<\|think\|>.*?(?:<\|/think\|>|$)", re.DOTALL)
_HIGHLIGHT_RE = re.compile(r"<\|highlight\|>.*?(?:<\|/highlight\|>|$)", re.DOTALL)
_FUNCTION_RE = re.compile(
    r"<function(?:=(?P<name>[^>\n]*))?>(?P<body>.*?)(?:</function>|$)",
    re.DOTALL,
)
_TOOL_RE = re.compile(r"<\|tool\|>(?P<body>.*?)(?:<\|/tool\|>|$)", re.DOTALL)
_PARAMETER_RE = re.compile(
    r"<parameter(?:=(?P<name>[^>\n]*))?>(?P<body>.*?)(?:</parameter>|$)",
    re.DOTALL,
)
_ORPHAN_CLOSE_RE = re.compile(
    r"</(?:function|parameter)>|<\|/(?:think|tool|highlight|canvas)\|>"
)
_HARNESS_MARKER_RE = re.compile(r"^\\?tau\s+task_index:", re.IGNORECASE)
_MULTI_NEWLINE_RE = re.compile(r"\n{3,}")


def _strip_decoration(content: str) -> str:
    content = _CANVAS_RE.sub("", content)
    content = _THINK_RE.sub("", content)
    content = _HIGHLIGHT_RE.sub("", content)
    content = _ORPHAN_CLOSE_RE.sub("", content)
    content = _MULTI_NEWLINE_RE.sub("\n\n", content)
    return content.strip()


def _parse_function_body(body: str) -> dict[str, Any]:
    args: dict[str, Any] = {}
    for match in _PARAMETER_RE.finditer(body or ""):
        key = (match.group("name") or "").strip() or f"arg_{len(args)}"
        raw = (match.group("body") or "").strip()
        try:
            args[key] = json.loads(raw)
        except json.JSONDecodeError:
            args[key] = raw
    leftover = _PARAMETER_RE.sub("", body or "").strip()
    if leftover and not args:
        try:
            parsed = json.loads(leftover)
            if isinstance(parsed, dict):
                return parsed
        except json.JSONDecodeError:
            return {"_raw": leftover}
    return args


def _expand_inline_tools(
    role: str, content: str, *, call_counter: list[int]
) -> list[Turn]:
    """Split assistant content that inlines <function> / <|tool|> into structured turns."""

    if role != "assistant" or (
        "<function" not in content and "<|tool|>" not in content
    ):
        text = _strip_decoration(content)
        return [{"role": role, "content": text}] if text else []

    turns: list[Turn] = []
    cursor = 0
    # Walk function then tool markers in order of appearance.
    pattern = re.compile(
        r"(?P<fn><function(?:=(?P<fname>[^>\n]*))?>(?P<fbody>.*?)(?:</function>|$))"
        r"|(?P<tool><\|tool\|>(?P<tbody>.*?)(?:<\|/tool\|>|$))",
        re.DOTALL,
    )
    for match in pattern.finditer(content):
        before = content[cursor : match.start()]
        before_text = _strip_decoration(before)
        if before_text:
            turns.append({"role": "assistant", "content": before_text})
        if match.group("fn") is not None:
            call_counter[0] += 1
            call_id = f"inline_{call_counter[0]}"
            name = (match.group("fname") or "").strip() or "unknown"
            turns.append(
                {
                    "role": "assistant",
                    "content": "",
                    "tool_calls": [
                        {
                            "id": call_id,
                            "name": name,
                            "arguments": _parse_function_body(
                                match.group("fbody") or ""
                            ),
                            "requestor": "assistant",
                        }
                    ],
                }
            )
        else:
            call_id = f"inline_{call_counter[0]}" if call_counter[0] else "inline_0"
            body = (match.group("tbody") or "").strip()
            turns.append(
                {
                    "role": "tool",
                    "tool_call_id": call_id,
                    "content": body,
                }
            )
        cursor = match.end()
    after = _strip_decoration(content[cursor:])
    if after:
        turns.append({"role": "assistant", "content": after})
    return turns


def normalize_conversation(messages: Any) -> list[Turn]:
    """Return tool-aware turns suitable for FMA review / LLM prompts.

    Prefer structured ``tool_calls`` / tool-role rows when present (new tau2 export).
    Otherwise expand τ-USI inline ``<function=…>`` / ``<|tool|>`` markup.
    """

    if not isinstance(messages, list):
        return []
    out: list[Turn] = []
    call_counter = [0]
    for msg in messages:
        if not isinstance(msg, dict):
            continue
        role = str(msg.get("role") or "").lower()
        if role in {"human", "customer"}:
            role = "user"
        elif role in {"agent", "bot", "chatbot", "model"}:
            role = "assistant"
        content = str(msg.get("content") or "")
        if role == "user" and _HARNESS_MARKER_RE.match(content.strip()):
            continue
        tool_calls = msg.get("tool_calls")
        if (
            isinstance(tool_calls, list)
            and tool_calls
            and not isinstance(tool_calls[0], bool)
        ):
            entry: Turn = {
                "role": role or "assistant",
                "content": _strip_decoration(content),
                "tool_calls": [],
            }
            for tc in tool_calls:
                if not isinstance(tc, dict):
                    continue
                entry["tool_calls"].append(
                    {
                        "id": str(tc.get("id") or ""),
                        "name": str(tc.get("name") or ""),
                        "arguments": tc.get("arguments")
                        if isinstance(tc.get("arguments"), dict)
                        else {},
                        "requestor": str(tc.get("requestor") or "assistant"),
                    }
                )
            out.append(entry)
            continue
        if role == "tool":
            out.append(
                {
                    "role": "tool",
                    "tool_call_id": str(msg.get("tool_call_id") or msg.get("id") or ""),
                    "content": content,
                    **({"error": True} if msg.get("error") else {}),
                }
            )
            continue
        if role not in {"user", "assistant", "system"}:
            continue
        out.extend(_expand_inline_tools(role, content, call_counter=call_counter))
    # Drop empty assistant turns with no tool_calls.
    cleaned: list[Turn] = []
    for turn in out:
        if (
            turn.get("role") == "assistant"
            and not turn.get("content")
            and not turn.get("tool_calls")
        ):
            continue
        cleaned.append(turn)
    return cleaned


def format_transcript(conversation: list[Turn], *, max_chars: int | None = None) -> str:
    """Numbered transcript for LLM prompts (includes tool calls/results)."""

    lines: list[str] = []
    for i, turn in enumerate(conversation):
        role = str(turn.get("role") or "").upper()
        content = str(turn.get("content") or "").strip()
        chunk = f"Turn {i} [{role}]"
        if turn.get("tool_calls"):
            calls = []
            for tc in turn["tool_calls"]:
                args = json.dumps(tc.get("arguments") or {}, ensure_ascii=False)
                calls.append(f"{tc.get('name')}({args}) id={tc.get('id')}")
            chunk += "\n  TOOL_CALLS: " + "; ".join(calls)
        if role == "TOOL":
            chunk += f" tool_call_id={turn.get('tool_call_id') or ''}"
            if turn.get("error"):
                chunk += " ERROR"
        if content:
            chunk += f"\n{content}"
        lines.append(chunk)
    text = "\n\n".join(lines)
    if max_chars is not None and len(text) > max_chars:
        return text[: max_chars - 20] + "\n...[truncated]..."
    return text
