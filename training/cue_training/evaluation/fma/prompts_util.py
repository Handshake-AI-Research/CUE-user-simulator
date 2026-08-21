"""Render jinja prompts under ``evaluation/fma/prompts/``."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Any

PROMPTS_DIR = Path(__file__).resolve().parent / "prompts"


@lru_cache(maxsize=16)
def _template(name: str) -> Any:
    try:
        from jinja2 import Environment, FileSystemLoader, select_autoescape
    except ImportError as exc:  # pragma: no cover
        raise ImportError(
            "jinja2 is required for FMA prompts; install with: uv sync --group fma"
        ) from exc
    env = Environment(
        loader=FileSystemLoader(str(PROMPTS_DIR)),
        autoescape=select_autoescape(enabled_extensions=()),
        trim_blocks=True,
        lstrip_blocks=True,
    )
    return env.get_template(name)


def render(name: str, **kwargs: Any) -> str:
    return _template(name).render(**kwargs)
