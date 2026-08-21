"""Apply human review decisions: write decisions.jsonl and update taxonomy examples."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any

from cue_training.evaluation.common.io import read_jsonl, write_jsonl
from cue_training.evaluation.fma.paths import batch_dir, tagged_path
from cue_training.evaluation.fma.store import (
    load_taxonomy,
    merge_modes,
    rename_mode,
    save_taxonomy,
    upsert_mode,
)


def _rewrite_label_field(
    row: dict[str, Any], mapping: Mapping[str, str], field: str
) -> bool:
    old = str(row.get(field) or "").strip()
    if not old or old not in mapping:
        return False
    row[field] = mapping[old]
    return True


def remap_labels(run_dir: Path, mapping: Mapping[str, str]) -> dict[str, int]:
    """Rewrite failure-mode labels across proposals/decisions/tagged using ``old -> new``."""

    clean = {
        str(k).strip(): str(v).strip()
        for k, v in mapping.items()
        if str(k).strip() and str(v).strip()
    }
    clean = {k: v for k, v in clean.items() if k != v}
    counts = {"proposals": 0, "decisions": 0, "tagged": 0}
    if not clean:
        return counts

    batches_root = run_dir / "batches"
    if batches_root.is_dir():
        for bdir in sorted(batches_root.iterdir()):
            if not bdir.is_dir():
                continue
            for name, key in (
                ("proposals.jsonl", "proposals"),
                ("decisions.jsonl", "decisions"),
            ):
                path = bdir / name
                if not path.is_file():
                    continue
                rows = list(read_jsonl(str(path)))
                changed = False
                for row in rows:
                    if _rewrite_label_field(row, clean, "label"):
                        counts[key] += 1
                        changed = True
                    if name == "decisions.jsonl" and _rewrite_label_field(
                        row, clean, "proposed_label"
                    ):
                        changed = True
                if changed:
                    write_jsonl(str(path), rows)

    tpath = tagged_path(run_dir)
    if tpath.is_file():
        rows = list(read_jsonl(str(tpath)))
        changed = False
        for row in rows:
            if _rewrite_label_field(row, clean, "label"):
                counts["tagged"] += 1
                changed = True
        if changed:
            write_jsonl(str(tpath), rows)
    return counts


def rename_failure_mode(
    run_dir: Path,
    *,
    old: str,
    new: str,
    description: str | None = None,
) -> dict[str, Any]:
    taxonomy = load_taxonomy(run_dir)
    rename_mode(taxonomy, old=old, new=new, description=description)
    save_taxonomy(run_dir, taxonomy)
    remapped = remap_labels(run_dir, {old: new})
    return {"taxonomy": taxonomy, "remapped": remapped}


def merge_failure_modes(
    run_dir: Path,
    *,
    sources: list[str],
    target: str,
    description: str | None = None,
) -> dict[str, Any]:
    taxonomy = load_taxonomy(run_dir)
    merge_modes(taxonomy, sources=sources, target=target, description=description)
    save_taxonomy(run_dir, taxonomy)
    mapping = {s: target for s in sources if s and s != target}
    remapped = remap_labels(run_dir, mapping)
    return {"taxonomy": taxonomy, "remapped": remapped}


def label_counts(run_dir: Path) -> dict[str, int]:
    """Count proposal + decision + tagged rows per failure-mode label across the run."""

    from collections import Counter

    counts: Counter[str] = Counter()
    batches_root = run_dir / "batches"
    if batches_root.is_dir():
        for bdir in sorted(batches_root.iterdir()):
            if not bdir.is_dir():
                continue
            for name in ("proposals.jsonl", "decisions.jsonl"):
                path = bdir / name
                if not path.is_file():
                    continue
                for row in read_jsonl(str(path)):
                    label = str(row.get("label") or "").strip()
                    if label:
                        counts[label] += 1
    tpath = tagged_path(run_dir)
    if tpath.is_file():
        for row in read_jsonl(str(tpath)):
            label = str(row.get("label") or "").strip()
            if label:
                counts[label] += 1
    # Also count unsaved review decisions? Not on disk — UI adds those separately.
    return dict(counts)


def prune_empty_modes(run_dir: Path, *, keep: set[str] | None = None) -> dict[str, Any]:
    """Drop taxonomy modes with no proposal/decision/tagged samples.

    Modes listed in ``keep`` are retained even with zero samples (e.g. Uncategorized).
    """

    taxonomy = load_taxonomy(run_dir)
    counts = label_counts(run_dir)
    keep = {*(keep or set()), "Uncategorized"}
    removed = [
        m["name"]
        for m in taxonomy
        if m.get("name")
        and m["name"] not in keep
        and counts.get(str(m["name"]), 0) == 0
    ]
    if removed:
        drop = set(removed)
        taxonomy = [m for m in taxonomy if m.get("name") not in drop]
        save_taxonomy(run_dir, taxonomy)
    return {
        "taxonomy": taxonomy,
        "removed": removed,
        "counts": {
            m["name"]: counts.get(str(m["name"]), 0) for m in taxonomy if m.get("name")
        },
    }


def finish_batch(
    run_dir: Path,
    batch: int,
    decisions: list[dict[str, Any]],
    *,
    max_examples: int = 5,
) -> dict[str, Any]:
    """Persist decisions and append accepted examples to taxonomy."""

    bdir = batch_dir(run_dir, batch)
    bdir.mkdir(parents=True, exist_ok=True)
    proposals = {
        str(r["primary_key"]): r
        for r in read_jsonl(str(bdir / "proposals.jsonl"))
        if r.get("primary_key")
    }
    sample = {
        str(r["primary_key"]): r
        for r in read_jsonl(str(bdir / "sample.jsonl"))
        if r.get("primary_key")
    }
    taxonomy = load_taxonomy(run_dir)
    normalized: list[dict[str, Any]] = []
    for dec in decisions:
        key = str(dec.get("primary_key") or "")
        if not key:
            continue
        prop = proposals.get(key) or {}
        row = sample.get(key) or {}
        label = str(dec.get("label") or prop.get("label") or "Uncategorized").strip()
        description = str(dec.get("description") or "").strip()
        explanation = str(
            dec.get("explanation") or prop.get("explanation") or ""
        ).strip()
        turn_indices = dec.get("turn_indices")
        if turn_indices is None:
            turn_indices = prop.get("turn_indices") or []
        excerpt = str(dec.get("excerpt") or prop.get("excerpt") or "").strip()
        notes = str(dec.get("notes") or "").strip()
        entry = {
            "primary_key": key,
            "episode_id": row.get("episode_id") or prop.get("episode_id"),
            "source_id": row.get("source_id") or prop.get("source_id"),
            "label": label,
            "explanation": explanation,
            "turn_indices": turn_indices,
            "excerpt": excerpt,
            "notes": notes,
            "proposed_label": prop.get("label"),
            "kept_or_changed": ("kept" if prop.get("label") == label else "changed"),
        }
        normalized.append(entry)
        upsert_mode(
            taxonomy,
            name=label,
            description=description or None,
            example={
                "episode_id": entry["episode_id"],
                "primary_key": key,
                "explanation": explanation or notes,
                "turn_indices": turn_indices,
                "snippet": excerpt,
            },
            max_examples=max_examples,
        )
    write_jsonl(str(bdir / "decisions.jsonl"), normalized)
    save_taxonomy(run_dir, taxonomy)
    n_tagged_updated = _apply_decisions_to_tagged(run_dir, normalized)
    return {
        "n_decisions": len(normalized),
        "n_tagged_updated": n_tagged_updated,
        "path": str(bdir / "decisions.jsonl"),
    }


def _apply_decisions_to_tagged(run_dir: Path, decisions: list[dict[str, Any]]) -> int:
    """Patch ``tagged.jsonl`` so TVD/plot pick up review labels without re-tagging."""

    path = tagged_path(run_dir)
    if not path.is_file() or not decisions:
        return 0
    by_key = {str(d["primary_key"]): d for d in decisions if d.get("primary_key")}
    rows = list(read_jsonl(str(path)))
    n = 0
    for row in rows:
        key = str(row.get("primary_key") or "")
        dec = by_key.get(key)
        if dec is None:
            continue
        row["label"] = dec.get("label") or row.get("label")
        row["explanation"] = dec.get("explanation") or row.get("explanation") or ""
        row["turn_indices"] = dec.get("turn_indices") or row.get("turn_indices") or []
        row["excerpt"] = dec.get("excerpt") or row.get("excerpt") or ""
        row["origin"] = "review"
        n += 1
    if n:
        write_jsonl(str(path), rows)
    return n
