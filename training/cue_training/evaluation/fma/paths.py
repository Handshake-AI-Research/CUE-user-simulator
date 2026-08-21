"""Run directory layout under ``$CUE_STORAGE_ROOT/fma/<run_id>/``."""

from __future__ import annotations

import os
from pathlib import Path

from cue_training.utils.config import storage_root


def default_fma_root() -> Path:
    return storage_root() / "fma"


def resolve_run_dir(run: str | Path) -> Path:
    """Accept an absolute/relative path or a run_id under ``$CUE_STORAGE_ROOT/fma``."""

    path = Path(run).expanduser()
    if (
        path.is_dir()
        or path.is_absolute()
        or "/" in str(run)
        or str(run).startswith(".")
    ):
        return path.resolve()
    return (default_fma_root() / str(run)).resolve()


def ensure_run(run_dir: Path) -> Path:
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "batches").mkdir(parents=True, exist_ok=True)
    return run_dir


def corpus_path(run_dir: Path) -> Path:
    return run_dir / "corpus.jsonl"


def taxonomy_path(run_dir: Path) -> Path:
    return run_dir / "taxonomy.json"


def tagged_path(run_dir: Path) -> Path:
    return run_dir / "tagged.jsonl"


def distributions_path(run_dir: Path) -> Path:
    return run_dir / "distributions.json"


def batch_dir(run_dir: Path, batch: int) -> Path:
    return run_dir / "batches" / f"{int(batch):02d}"


def next_batch_index(run_dir: Path) -> int:
    root = run_dir / "batches"
    if not root.is_dir():
        return 1
    idxs = []
    for child in root.iterdir():
        if child.is_dir() and child.name.isdigit():
            idxs.append(int(child.name))
    return (max(idxs) + 1) if idxs else 1


def default_model() -> str:
    return (
        os.environ.get("FMA_MODEL")
        or os.environ.get("OPENAI_MODEL")
        or "claude-fable-5"
    )


def api_key_env_for_model(model: str) -> str:
    """Pick the credential env var LiteLLM should use for ``model``."""

    m = (model or "").lower()
    if any(tag in m for tag in ("claude", "anthropic", "fable")):
        return "ANTHROPIC_API_KEY"
    if "gemini" in m or m.startswith("google/"):
        return "GOOGLE_API_KEY"
    if "openrouter" in m:
        return "OPENROUTER_API_KEY"
    return "OPENAI_API_KEY"
