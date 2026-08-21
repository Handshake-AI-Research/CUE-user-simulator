"""Sample failed trajectories for an FMA audit batch."""

from __future__ import annotations

import random
from pathlib import Path
from typing import Any

from cue_training.evaluation.common.io import read_jsonl, write_jsonl
from cue_training.evaluation.fma.paths import (
    batch_dir,
    corpus_path,
    ensure_run,
    tagged_path,
)
from cue_training.evaluation.fma.store import index_by_key


def _already_reviewed(run_dir: Path) -> set[str]:
    keys: set[str] = set()
    batches = run_dir / "batches"
    if not batches.is_dir():
        return keys
    for child in batches.iterdir():
        decisions = child / "decisions.jsonl"
        if not decisions.is_file():
            continue
        for row in read_jsonl(str(decisions)):
            if row.get("primary_key"):
                keys.add(str(row["primary_key"]))
    return keys


def sample_batch(
    run_dir: Path,
    *,
    batch: int,
    n: int = 100,
    seed: int = 0,
) -> dict[str, Any]:
    ensure_run(run_dir)
    corpus = read_jsonl(str(corpus_path(run_dir)))
    done = _already_reviewed(run_dir)
    pool = [r for r in corpus if str(r.get("primary_key")) not in done]
    rng = random.Random(int(seed) + int(batch) * 1_000_003)
    if len(pool) <= n:
        chosen = list(pool)
        rng.shuffle(chosen)
    else:
        chosen = rng.sample(pool, n)
    out = batch_dir(run_dir, batch)
    out.mkdir(parents=True, exist_ok=True)
    write_jsonl(str(out / "sample.jsonl"), chosen)
    return {
        "batch": batch,
        "n_sampled": len(chosen),
        "n_pool": len(pool),
        "path": str(out / "sample.jsonl"),
    }


def sample_uncategorized_batch(
    run_dir: Path,
    *,
    batch: int,
    n: int | None = None,
    seed: int = 0,
    label: str = "Uncategorized",
) -> dict[str, Any]:
    """Build a review batch from auto-tagged ``Uncategorized`` failures.

    Joins ``tagged.jsonl`` with the corpus so the existing review UI can relabel
    them; skips keys already present in any ``decisions.jsonl``.
    """

    ensure_run(run_dir)
    tagged_file = tagged_path(run_dir)
    if not tagged_file.is_file():
        raise FileNotFoundError(
            f"no tagged.jsonl under {run_dir}; run cue-fma tag first"
        )
    corpus_by_key = load_corpus_index(run_dir)
    done = _already_reviewed(run_dir)
    target = (label or "Uncategorized").strip() or "Uncategorized"
    pool: list[tuple[dict[str, Any], dict[str, Any]]] = []
    for tag in read_jsonl(str(tagged_file)):
        key = str(tag.get("primary_key") or "")
        if not key or key in done:
            continue
        if str(tag.get("label") or "").strip() != target:
            continue
        row = corpus_by_key.get(key)
        if row is None:
            continue
        pool.append((row, tag))
    rng = random.Random(int(seed) + int(batch) * 1_000_003)
    rng.shuffle(pool)
    n_pool = len(pool)
    if n is not None:
        pool = pool[: max(0, int(n))]
    out = batch_dir(run_dir, batch)
    out.mkdir(parents=True, exist_ok=True)
    samples = [row for row, _ in pool]
    proposals = [
        {
            "primary_key": str(tag.get("primary_key") or row.get("primary_key")),
            "episode_id": tag.get("episode_id") or row.get("episode_id"),
            "source_id": tag.get("source_id") or row.get("source_id"),
            "label": tag.get("label") or target,
            "explanation": tag.get("explanation") or "",
            "turn_indices": tag.get("turn_indices") or [],
            "excerpt": tag.get("excerpt") or "",
            "origin": "tag",
        }
        for row, tag in pool
    ]
    write_jsonl(str(out / "sample.jsonl"), samples)
    write_jsonl(str(out / "proposals.jsonl"), proposals)
    return {
        "batch": batch,
        "n_sampled": len(samples),
        "n_pool": n_pool,
        "label": target,
        "path": str(out / "sample.jsonl"),
    }


def load_sample(run_dir: Path, batch: int) -> list[dict[str, Any]]:
    path = batch_dir(run_dir, batch) / "sample.jsonl"
    return read_jsonl(str(path))


def load_corpus_index(run_dir: Path) -> dict[str, dict[str, Any]]:
    return index_by_key(read_jsonl(str(corpus_path(run_dir))))
