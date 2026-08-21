"""Rehydrate a published CUE annotation delta into joint-training JSONL files."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Sequence

from cue_training.data.annotation_release import (
    coverage_report,
    existing_record_ids,
    load_delta_from_hub,
    rehydrate_delta_rows,
    write_rehydrated_dataset,
)
from cue_training.utils.config import storage_root
from cue_training.utils.log import log

TAG = "hydrate_annotations"


def hydrate_annotations(
    *,
    repo_id: str,
    out_root: str | Path | None = None,
    config_name: str | None = None,
    splits: Sequence[str] | None = None,
    revision: str | None = None,
    token: str | None = None,
    max_records_per_source: int | None = None,
    min_coverage: float = 0.95,
    resume: bool = True,
) -> Path:
    """Download delta annotations and reconstruct the training-record layout."""
    output = Path(out_root or (storage_root() / "annotations" / "profiled_with_style"))
    rows = load_delta_from_hub(
        repo_id,
        config_name=config_name,
        split=list(splits) if splits else None,
        revision=revision,
        token=token,
    )
    if not rows:
        raise ValueError(f"no annotation rows found in {repo_id}")
    pending = rows
    # Stand-ins for records already on disk, so they still count toward coverage.
    kept: list[dict[str, str]] = []
    if resume:
        have = existing_record_ids(output)
        pending = []
        for row in rows:
            key = (str(row["dataset"]), str(row.get("split") or "train"))
            if str(row["source_id"]) in have.get(key, set()):
                kept.append({"dataset": row["dataset"], "id": str(row["source_id"])})
            else:
                pending.append(row)
        log(TAG, f"resume: keeping {len(kept)} existing records, hydrating {len(pending)}")

    records, errors = (
        rehydrate_delta_rows(pending, max_records_per_source=max_records_per_source)
        if pending
        else ([], [])
    )
    report = coverage_report(rows, [*records, *kept], errors)
    output.mkdir(parents=True, exist_ok=True)
    audit_path = output / "hydration_audit.json"
    audit_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    for dataset, item in report["sources"].items():
        log(TAG, f"{dataset}: hydrated {item['hydrated']}/{item['requested']} ({item['coverage']:.1%})")
    short = {
        dataset: item["coverage"]
        for dataset, item in report["sources"].items()
        if item["coverage"] < min_coverage
    }
    if short:
        raise ValueError(
            f"hydration coverage below {min_coverage:.0%} for "
            f"{', '.join(f'{k}={v:.1%}' for k, v in sorted(short.items()))}; "
            f"nothing written. See {audit_path}"
        )
    written = write_rehydrated_dataset(records, output, append=resume)
    log(TAG, f"rehydrated {len(records)} records across {len(written)} files -> {output}")
    if errors:
        log(TAG, f"skipped {len(errors)} rows that could not be rehydrated")
    return output
