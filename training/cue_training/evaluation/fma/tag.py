"""Tag remaining corpus failures with the frozen taxonomy."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from cue_training.baselines.common.llm import accumulate_chat_cost
from cue_training.evaluation.common.io import read_jsonl, write_jsonl
from cue_training.evaluation.fma.paths import tagged_path
from cue_training.evaluation.fma.propose import (
    DEFAULT_WORKERS,
    _map_concurrent,
    classify_one,
)
from cue_training.evaluation.fma.store import load_taxonomy, upsert_mode


def _reviewed_keys(run_dir: Path) -> dict[str, dict[str, Any]]:
    out: dict[str, dict[str, Any]] = {}
    batches = run_dir / "batches"
    if not batches.is_dir():
        return out
    for child in sorted(batches.iterdir()):
        path = child / "decisions.jsonl"
        if not path.is_file():
            continue
        for row in read_jsonl(str(path)):
            key = str(row.get("primary_key") or "")
            if key:
                out[key] = {
                    "primary_key": key,
                    "episode_id": row.get("episode_id"),
                    "source_id": row.get("source_id"),
                    "label": row.get("label"),
                    "explanation": row.get("notes") or row.get("explanation") or "",
                    "turn_indices": row.get("turn_indices") or [],
                    "excerpt": row.get("excerpt") or "",
                    "origin": "review",
                }
    return out


def _partial_path(run_dir: Path) -> Path:
    return run_dir / "tagged.partial.jsonl"


def _prior_tagged(run_dir: Path) -> dict[str, dict[str, Any]]:
    """Existing auto/review tags from a previous ``cue-fma tag`` (keyed by primary_key)."""

    out: dict[str, dict[str, Any]] = {}
    # The partial file holds calls that completed in a pass that was interrupted
    # before the final write, so a killed run resumes instead of re-paying for them.
    for path in (tagged_path(run_dir), _partial_path(run_dir)):
        if not path.is_file():
            continue
        for row in read_jsonl(str(path)):
            key = str(row.get("primary_key") or "")
            if key:
                out[key] = row
    return out


def tag_remainder(
    run_dir: Path,
    *,
    model: str | None = None,
    chat_fn: Any = None,
    workers: int = DEFAULT_WORKERS,
    force: bool = False,
    relabel: list[str] | None = None,
    reasoning_effort: str | None = None,
) -> dict[str, Any]:
    from cue_training.baselines.common.llm import chat as default_chat
    from cue_training.evaluation.fma.paths import corpus_path, default_model

    taxonomy = load_taxonomy(run_dir)
    upsert_mode(taxonomy, name="Uncategorized", description="No taxonomy mode fits.")
    corpus = read_jsonl(str(corpus_path(run_dir)))
    relabel_set = {str(x).strip() for x in (relabel or []) if str(x).strip()}
    reviewed = _reviewed_keys(run_dir)
    # Resume from prior tagged.jsonl unless --force: only classify keys never labeled.
    # --relabel drops matching prior labels (auto + review) so those rows are reclassified.
    kept: dict[str, dict[str, Any]] = {} if force else dict(_prior_tagged(run_dir))
    if relabel_set and not force:
        kept = {
            k: v
            for k, v in kept.items()
            if str(v.get("label") or "").strip() not in relabel_set
        }
        reviewed = {
            k: v
            for k, v in reviewed.items()
            if str(v.get("label") or "").strip() not in relabel_set
        }
    # Review decisions always win over a stale auto-tag for the same key.
    kept.update(reviewed)
    seen = set(kept)
    fn = chat_fn or default_chat
    model_name = model or default_model()
    remainder = [
        row
        for row in corpus
        if (key := str(row.get("primary_key") or "")) and key not in seen
    ]
    partial = _partial_path(run_dir)
    with (
        accumulate_chat_cost() as cost,
        partial.open("w" if force else "a", encoding="utf-8") as ckpt,
    ):

        def _checkpoint(row: dict[str, Any] | None) -> None:
            if row is None:
                return
            ckpt.write(json.dumps(row, ensure_ascii=False) + "\n")
            ckpt.flush()

        def _classify(row: dict[str, Any]) -> dict[str, Any] | None:
            try:
                return classify_one(
                    row,
                    taxonomy=taxonomy,
                    model=model_name,
                    chat_fn=fn,
                    reasoning_effort=reasoning_effort,
                )
            except Exception as exc:  # noqa: BLE001
                # Leave the row untagged so the next pass retries it, rather than
                # losing a multi-hour run to one row that exhausted its retries.
                print(
                    f"[cue-fma tag] {type(exc).__name__} on {row.get('primary_key')}: {exc}",
                    flush=True,
                )
                return None

        results = _map_concurrent(
            _classify,
            remainder,
            workers=workers,
            desc="cue-fma tag",
            on_result=_checkpoint,
        )
    classified = [row for row in results if row]
    print(cost.format_line(label="cue-fma tag"), flush=True)
    for row in classified:
        key = str(row.get("primary_key") or "")
        if key:
            kept[key] = row
    # Align with the current corpus (drop labels for failures removed on reinjest).
    tagged = [
        kept[key]
        for row in corpus
        if (key := str(row.get("primary_key") or "")) in kept
    ]
    write_jsonl(str(tagged_path(run_dir)), tagged)
    partial.unlink(missing_ok=True)
    return {
        "n_corpus": len(corpus),
        "n_tagged": len(tagged),
        "n_from_review": len(reviewed),
        "n_reused": len(tagged) - len(classified),
        "n_classified": len(classified),
        "n_failed": len(results) - len(classified),
        "workers": max(1, min(int(workers), max(1, len(remainder) or 1))),
        "force": bool(force),
        "relabel": sorted(relabel_set),
        "model": model_name,
        "cost": cost.summary(),
    }
