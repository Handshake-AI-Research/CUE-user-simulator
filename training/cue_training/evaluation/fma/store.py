"""JSON/JSONL helpers and taxonomy I/O for an FMA run."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from cue_training.evaluation.common.io import read_jsonl, write_jsonl
from cue_training.evaluation.fma.paths import taxonomy_path


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )


def load_taxonomy(run_dir: Path) -> list[dict[str, Any]]:
    path = taxonomy_path(run_dir)
    if not path.is_file():
        return []
    data = read_json(path)
    if isinstance(data, list):
        return list(data)
    return list(data.get("modes") or [])


def save_taxonomy(run_dir: Path, modes: list[dict[str, Any]]) -> None:
    write_json(taxonomy_path(run_dir), modes)


def empty_mode(name: str, description: str = "") -> dict[str, Any]:
    return {"name": name, "description": description or "", "examples": []}


def upsert_mode(
    modes: list[dict[str, Any]],
    *,
    name: str,
    description: str | None = None,
    example: dict[str, Any] | None = None,
    max_examples: int = 5,
) -> list[dict[str, Any]]:
    for mode in modes:
        if mode.get("name") == name:
            if description is not None and description.strip():
                mode["description"] = description.strip()
            if example is not None:
                examples = list(mode.get("examples") or [])
                examples.append(example)
                mode["examples"] = examples[-max_examples:]
            return modes
    mode = empty_mode(name, description or "")
    if example is not None:
        mode["examples"] = [example]
    modes.append(mode)
    return modes


def rename_mode(
    modes: list[dict[str, Any]],
    *,
    old: str,
    new: str,
    description: str | None = None,
) -> list[dict[str, Any]]:
    """Rename a taxonomy mode in place. No-op if ``old`` is missing."""

    old = (old or "").strip()
    new = (new or "").strip()
    if not old or not new or old == new:
        return modes
    src = next((m for m in modes if m.get("name") == old), None)
    if src is None:
        return modes
    dst = next((m for m in modes if m.get("name") == new), None)
    if dst is None:
        src["name"] = new
        if description is not None and description.strip():
            src["description"] = description.strip()
        return modes
    # Target already exists: merge examples into it and drop the old name.
    return merge_modes(modes, sources=[old], target=new, description=description)


def merge_modes(
    modes: list[dict[str, Any]],
    *,
    sources: list[str],
    target: str,
    description: str | None = None,
    max_examples: int = 5,
) -> list[dict[str, Any]]:
    """Merge ``sources`` into ``target`` (creating target if needed); drop sources."""

    target = (target or "").strip()
    src_names = [
        s.strip() for s in sources if (s or "").strip() and s.strip() != target
    ]
    if not target or not src_names:
        return modes
    by_name = {str(m.get("name") or ""): m for m in modes if m.get("name")}
    dst = by_name.get(target)
    if dst is None:
        dst = empty_mode(target, description or "")
        modes.append(dst)
        by_name[target] = dst
    elif description is not None and description.strip():
        dst["description"] = description.strip()
    examples = list(dst.get("examples") or [])
    drop: set[str] = set()
    for name in src_names:
        src = by_name.get(name)
        if src is None:
            continue
        examples.extend(list(src.get("examples") or []))
        if not dst.get("description") and src.get("description"):
            dst["description"] = src["description"]
        drop.add(name)
    dst["examples"] = examples[-max_examples:]
    if drop:
        modes[:] = [m for m in modes if m.get("name") not in drop]
    return modes


def index_by_key(
    rows: list[dict[str, Any]], key: str = "primary_key"
) -> dict[str, dict[str, Any]]:
    return {str(r[key]): r for r in rows if r.get(key) is not None}


__all__ = [
    "empty_mode",
    "index_by_key",
    "load_taxonomy",
    "merge_modes",
    "read_json",
    "read_jsonl",
    "rename_mode",
    "save_taxonomy",
    "upsert_mode",
    "write_json",
    "write_jsonl",
]
