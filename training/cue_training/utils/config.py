"""YAML/JSON config loading with dotted-key access."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any


def storage_root() -> Path:
    """Return the shared root for data, models, outputs, and caches."""

    default = Path(__file__).resolve().parents[2] / "artifacts"
    return Path(os.environ.get("CUE_STORAGE_ROOT", default)).expanduser().resolve()


def expand_config_text(text: str, path: str | Path) -> str:
    """Expand environment variables, with a repo-local CUE storage default."""

    root = str(storage_root())
    text = text.replace("${CUE_STORAGE_ROOT}", root).replace(
        "$CUE_STORAGE_ROOT", root
    )
    return os.path.expandvars(text)


def load_config(path: str | Path) -> dict[str, Any]:
    path = Path(path)
    text = expand_config_text(path.read_text(encoding="utf-8"), path)
    if path.suffix in {".yaml", ".yml"}:
        try:
            import yaml  # type: ignore
        except ImportError as exc:
            raise ImportError(
                "PyYAML is required for .yaml configs. Install pyyaml or use JSON."
            ) from exc
        data = yaml.safe_load(text)
    else:
        data = json.loads(text)
    if not isinstance(data, dict):
        raise ValueError(f"Config must be a mapping: {path}")
    return data


def get(cfg: dict[str, Any], key: str, default: Any = None) -> Any:
    cur: Any = cfg
    for part in key.split("."):
        if not isinstance(cur, dict) or part not in cur:
            return default
        cur = cur[part]
    return cur
