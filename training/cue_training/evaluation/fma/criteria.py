"""Load tau2-bench task success criteria for FMA prompts."""

from __future__ import annotations

import json
from collections.abc import Mapping
from functools import lru_cache
from pathlib import Path
from typing import Any


def _clean(val: Any) -> str:
    if val is None:
        return ""
    s = str(val).strip()
    return "" if s.lower() == "nan" else s


def _format_action(action: Mapping[str, Any]) -> str:
    name = _clean(action.get("name")) or "unknown_action"
    arguments = action.get("arguments")
    if isinstance(arguments, Mapping) and arguments:
        args = ", ".join(
            f"{k}={json.dumps(v, ensure_ascii=False)}" for k, v in arguments.items()
        )
        call = f"{name}({args})"
    else:
        call = f"{name}()"
    return call


def criteria_text(task: Mapping[str, Any]) -> str:
    criteria = task.get("evaluation_criteria") or {}
    lines: list[str] = []
    purpose = _clean((task.get("description") or {}).get("purpose"))
    if purpose:
        lines.append(f"What this task is testing: {purpose}")
    for label, key in (
        ("The agent is required to", "nl_assertions"),
        ("The agent must communicate to the user", "communicate_info"),
    ):
        items = [_clean(i) for i in (criteria.get(key) or [])]
        items = [i for i in items if i]
        if items:
            lines.append(label + ":")
            lines.extend(f"- {i}" for i in items)
    actions = [
        _format_action(a) for a in (criteria.get("actions") or []) if _format_action(a)
    ]
    if actions:
        lines.append("Required backend actions (expected tool calls / state changes):")
        lines.extend(f"- {a}" for a in actions)
    return "\n".join(lines)


def default_tasks_dir() -> Path:
    return (
        Path(__file__).resolve().parents[1]
        / "tau2_bench"
        / "external"
        / "tau2-bench"
        / "data"
        / "tau2"
        / "domains"
    )


@lru_cache(maxsize=4)
def load_task_criteria(tasks_dir: str | None = None) -> dict[tuple[str, str], str]:
    root = Path(tasks_dir).expanduser() if tasks_dir else default_tasks_dir()
    if not root.is_absolute():
        root = Path.cwd() / root
    if not root.is_dir():
        return {}
    out: dict[tuple[str, str], str] = {}
    for path in sorted(root.glob("*/tasks.json")):
        try:
            tasks = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        for task in tasks:
            text = criteria_text(task)
            if text:
                out[(path.parent.name, _clean(task.get("id")))] = text
    return out
