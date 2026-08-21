"""Annotation-delta export / rehydrate helpers (IDs + hashes + manuals only)."""

from __future__ import annotations

import json
from collections.abc import Iterable, Iterator, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from cue_training.data.schema import ManualValidationError, validate_manual, validate_record
from cue_training.data.streaming import discover_jsonl_files, iter_jsonl, write_jsonl_atomic, write_streaming_index
from cue_training.utils.log import log, warn

TAG = "annotation_release"
ANNOTATION_VERSION = 1

# Official DialogStudio config names (task-oriented + casing helpers).
# Source: https://huggingface.co/datasets/Salesforce/dialogstudio
DIALOGSTUDIO_DATASETS = frozenset(
    {
        "ABCD",
        "AirDialogue",
        "BiTOD",
        "CaSiNo",
        "CraigslistBargains",
        "Disambiguation",
        "DSTC2-Clean",
        "FRAMES",
        "GECOR",
        "HDSA-Dialog",
        "KETOD",
        "KVRET",
        "MetaLWOZ",
        "MS-DC",
        "MuDoCo",
        "MulDoGO",
        "MultiWOZ_2.1",
        "MULTIWOZ2_2",
        "SGD",
        "SimJointGEN",
        "SimJointMovie",
        "SimJointRestaurant",
        "STAR",
        "Taskmaster1",
        "Taskmaster2",
        "Taskmaster3",
        "WOZ2_0",
    }
)

_PROVENANCE_KEEP = frozenset(
    {
        "proposer",
        "sim_models",
        "max_user_turns",
        "exclude_sim",
        "annotation_version",
    }
)

_DELTA_REQUIRED = (
    "dataset",
    "split",
    "source_repo",
    "source_split",
    "source_id",
    "persona_manual",
    "annotation_version",
)

DIALOGSTUDIO_REPO = "Salesforce/dialogstudio"
LMSYS_REPO = "lmsys/lmsys-chat-1m"
WILDCHAT_REPO = "allenai/WildChat-1M"


@dataclass(frozen=True)
class SourceSpec:
    repo: str
    config: str | None
    split: str
    kind: str  # dialogstudio | lmsys | wildchat


_DIALOGSTUDIO_BY_LOWER = {name.lower(): name for name in DIALOGSTUDIO_DATASETS}

# Historical / shorthand dataset directory names seen in annotated JSONL.
_DATASET_ALIASES = {
    "lmsys": "lmsys-chat-1m",
    "lmsys-chat-1m": "lmsys-chat-1m",
    "lmsys_chat_1m": "lmsys-chat-1m",
    "lmsys-chat": "lmsys-chat-1m",
    "wildchat": "wildchat",
    "wildchat-1m": "wildchat",
    "wildchat_1m": "wildchat",
    # Common DialogStudio spelling variants.
    "multiwoz2.1": "MultiWOZ_2.1",
    "multiwoz_2_1": "MultiWOZ_2.1",
    "multiwoz2_2": "MULTIWOZ2_2",
    "multiwoz_2.2": "MULTIWOZ2_2",
    "woz2.0": "WOZ2_0",
    "woz_2_0": "WOZ2_0",
    "dstc2_clean": "DSTC2-Clean",
    "dstc2-clean": "DSTC2-Clean",
}


def canonical_dataset(dataset: str) -> str:
    """Map a dataset directory / field name onto its canonical release name."""

    name = str(dataset or "").strip()
    if not name:
        return name
    if name in DIALOGSTUDIO_DATASETS:
        return name
    lowered = name.lower()
    if lowered in _DATASET_ALIASES:
        return _DATASET_ALIASES[lowered]
    if lowered in _DIALOGSTUDIO_BY_LOWER:
        return _DIALOGSTUDIO_BY_LOWER[lowered]
    return name


def resolve_source(dataset: str, *, split: str = "train") -> SourceSpec:
    """Map a CUE dataset name to its upstream HF source.

    Everything except LMSYS / WildChat rehydrates from Salesforce/dialogstudio
    using the (canonical) dataset name as the DialogStudio config.
    """

    name = canonical_dataset(dataset)
    if not name:
        raise ValueError("dataset name is required")
    if name == "lmsys-chat-1m":
        return SourceSpec(LMSYS_REPO, None, split, "lmsys")
    if name == "wildchat":
        return SourceSpec(WILDCHAT_REPO, None, split, "wildchat")
    return SourceSpec(DIALOGSTUDIO_REPO, name, split, "dialogstudio")


def released_persona_manual(manual: Any) -> dict[str, Any]:
    """Canonical released annotation schema (validated persona_manual)."""

    return validate_manual(manual)


def sanitize_provenance(provenance: Mapping[str, Any] | None) -> dict[str, Any]:
    if not isinstance(provenance, Mapping):
        return {"annotation_version": ANNOTATION_VERSION}
    out = {k: provenance[k] for k in _PROVENANCE_KEEP if k in provenance}
    out["annotation_version"] = int(provenance.get("annotation_version") or ANNOTATION_VERSION)
    return out


def source_native_id(dataset: str, source_id: str, metadata: Mapping[str, Any] | None = None) -> str:
    """Best-effort native source key used during rehydration lookups."""

    meta = metadata or {}
    for key in ("conversation_hash", "conversation_id", "source_native_id"):
        value = meta.get(key)
        if value not in (None, ""):
            return str(value)
    sid = str(source_id or "").strip()
    if canonical_dataset(dataset) == "wildchat" and "-" in sid:
        base, suffix = sid.rsplit("-", 1)
        if suffix.isdigit() and base:
            return base
    return sid


def build_delta_row(record: Mapping[str, Any], *, source_revision: str | None = None) -> dict[str, Any]:
    """Build one private HF delta row from an annotated training record."""

    validated = validate_record(dict(record))
    if not validated.get("persona_manual"):
        raise ManualValidationError("annotated record requires persona_manual")
    dataset = canonical_dataset(validated.get("dataset") or record.get("dataset") or "")
    if not dataset:
        raise ValueError("record requires dataset")
    split = str(validated.get("split") or "train")
    spec = resolve_source(dataset, split=split)
    source_id = str(validated["id"])
    meta = dict(validated.get("metadata") or {})
    # Keep only non-conversation metadata keys that help attribution / filters.
    keep_meta = {
        k: meta[k]
        for k in ("language", "model", "timestamp", "conversation_hash", "conversation_id")
        if meta.get(k) not in (None, "")
    }
    row = {
        "dataset": dataset,
        "split": split,
        "source_repo": spec.repo,
        "source_config": spec.config,
        "source_split": spec.split,
        "source_revision": source_revision,
        "source_id": source_id,
        "source_native_id": source_native_id(dataset, source_id, meta),
        "persona_manual": released_persona_manual(validated["persona_manual"]),
        "annotation_version": ANNOTATION_VERSION,
        "provenance": sanitize_provenance(validated.get("provenance")),
        "metadata": keep_meta,
    }
    return row


def validate_delta_row(row: Any) -> dict[str, Any]:
    if not isinstance(row, dict):
        raise ValueError("delta row must be an object")
    missing = [k for k in _DELTA_REQUIRED if k not in row]
    if missing:
        raise ValueError(f"delta row missing fields: {missing}")
    manual = released_persona_manual(row["persona_manual"])
    dataset = canonical_dataset(row["dataset"])
    split = str(row.get("split") or "train")
    spec = resolve_source(dataset, split=split)
    source_id = str(row["source_id"]).strip()
    if not source_id:
        raise ValueError("delta row requires source_id")
    return {
        "dataset": dataset,
        "split": split,
        # Always remap through the registry so DialogStudio / LMSYS / WildChat
        # sources are canonical even if an older delta stored a different repo.
        "source_repo": spec.repo,
        "source_config": spec.config,
        "source_split": str(row.get("source_split") or spec.split),
        "source_revision": row.get("source_revision"),
        "source_id": source_id,
        "source_native_id": str(row.get("source_native_id") or source_native_id(dataset, source_id)),
        "persona_manual": manual,
        "annotation_version": int(row.get("annotation_version") or ANNOTATION_VERSION),
        "provenance": sanitize_provenance(row.get("provenance")),
        "metadata": dict(row.get("metadata") or {}),
    }


def iter_annotated_records(root: str | Path, *, split: str = "train") -> Iterator[dict[str, Any]]:
    root = Path(root)
    for path in discover_jsonl_files(root, split=split):
        dataset = path.parent.name
        for raw in iter_jsonl(path, validate=False):
            if not isinstance(raw, dict):
                continue
            raw = {**raw, "dataset": raw.get("dataset") or dataset, "split": raw.get("split") or split}
            try:
                yield validate_record(raw)
            except (ValueError, TypeError) as exc:
                warn(TAG, f"skip invalid annotated record {path}: {exc}")


def export_delta_rows(
    annotated_root: str | Path,
    *,
    split: str = "train",
    source_revision: str | None = None,
    datasets: Sequence[str] | None = None,
) -> list[dict[str, Any]]:
    allow = {d.strip() for d in datasets} if datasets else None
    rows: list[dict[str, Any]] = []
    skipped = 0
    for record in iter_annotated_records(annotated_root, split=split):
        dataset = str(record.get("dataset") or "")
        if allow is not None and dataset not in allow:
            continue
        if not record.get("persona_manual"):
            skipped += 1
            continue
        try:
            rows.append(build_delta_row(record, source_revision=source_revision))
        except (ValueError, ManualValidationError) as exc:
            skipped += 1
            warn(TAG, f"skip export id={record.get('id')}: {exc}")
    log(TAG, f"exported {len(rows)} delta rows split={split} (skipped={skipped})")
    return rows


def export_delta_rows_multi(
    roots_by_split: Mapping[str, str | Path],
    *,
    source_revision: str | None = None,
    datasets: Sequence[str] | None = None,
) -> list[dict[str, Any]]:
    """Export delta rows from one or more annotated roots keyed by split name."""

    rows: list[dict[str, Any]] = []
    for split, root in roots_by_split.items():
        if root is None:
            continue
        path = Path(root)
        if not path.is_dir():
            warn(TAG, f"skip missing annotated root split={split} path={path}")
            continue
        rows.extend(
            export_delta_rows(
                path,
                split=str(split),
                source_revision=source_revision,
                datasets=datasets,
            )
        )
    return rows


def pin_source_revisions(
    rows: Sequence[Mapping[str, Any]],
    *,
    token: str | None = None,
) -> list[dict[str, Any]]:
    """Stamp each row with the current upstream commit sha of its source repo."""

    from huggingface_hub import HfApi

    api = HfApi(token=token)
    shas: dict[str, str] = {}
    out: list[dict[str, Any]] = []
    for row in rows:
        repo = str(row["source_repo"])
        if repo not in shas:
            shas[repo] = api.dataset_info(repo).sha
            log(TAG, f"pinned {repo} -> {shas[repo]}")
        out.append({**row, "source_revision": shas[repo]})
    return out


def write_delta_jsonl(path: str | Path, rows: Iterable[Mapping[str, Any]]) -> int:
    validated = [validate_delta_row(row) for row in rows]
    return write_jsonl_atomic(path, validated)


def load_delta_jsonl(path: str | Path) -> list[dict[str, Any]]:
    return [validate_delta_row(row) for row in iter_jsonl(path, validate=False)]


def _flatten_delta_row(row: Mapping[str, Any]) -> dict[str, Any]:
    validated = validate_delta_row(row)
    return {
        **validated,
        "source_config": validated.get("source_config"),
        "persona_manual": json.dumps(validated["persona_manual"], ensure_ascii=False),
        "provenance": json.dumps(validated.get("provenance") or {}, ensure_ascii=False),
        "metadata": json.dumps(validated.get("metadata") or {}, ensure_ascii=False),
    }


def group_delta_by_dataset_split(
    rows: Sequence[Mapping[str, Any]],
) -> dict[str, dict[str, list[dict[str, Any]]]]:
    """Group flat delta rows into `{dataset: {split: [rows...]}}`."""

    grouped: dict[str, dict[str, list[dict[str, Any]]]] = {}
    for raw in rows:
        flat = _flatten_delta_row(raw)
        dataset = str(flat["dataset"])
        split = str(flat.get("split") or "train")
        grouped.setdefault(dataset, {}).setdefault(split, []).append(flat)
    return grouped


_DIALOGSTUDIO_CATEGORY = "task_oriented"
_DIALOGSTUDIO_SPLIT_DIRS = {"validation": "val"}


def iter_dialogstudio_rows(spec: SourceSpec, *, revision: str | None = None) -> Iterator[dict[str, Any]]:
    """Yield DialogStudio dialogs straight from the repo's JSON files.

    DialogStudio ships a loading script and no parquet, and ``datasets`` dropped
    script support in 4.x, so ``load_dataset`` cannot read it at all. The rows are
    rebuilt the way ``dialogstudio.py`` did: files merged in numeric order, with
    the JSON object key (``SGD--train--1``) as the dialog id the delta stores.
    """

    from huggingface_hub import hf_hub_download, list_repo_files

    split_dir = _DIALOGSTUDIO_SPLIT_DIRS.get(spec.split, spec.split)
    prefix = f"{_DIALOGSTUDIO_CATEGORY}/{spec.config}/{split_dir}/dialogues_"
    names = [
        name
        for name in list_repo_files(spec.repo, repo_type="dataset", revision=revision)
        if name.startswith(prefix) and name.endswith(".json")
    ]
    if not names:
        raise ValueError(f"no DialogStudio files for config={spec.config} split={spec.split}")
    names.sort(key=lambda name: int(name.rsplit("_", 1)[1].removesuffix(".json")))
    for name in names:
        path = hf_hub_download(spec.repo, name, repo_type="dataset", revision=revision)
        with open(path, encoding="utf-8") as handle:
            dialogs = json.load(handle)
        for dialog_id, dialog in dialogs.items():
            entries = dialog.get("log")
            yield {
                **dialog,
                "new dialog id": dialog_id,
                "log": [entries] if isinstance(entries, Mapping) else (entries or []),
            }


def load_source_records(
    spec: SourceSpec,
    *,
    revision: str | None = None,
    max_records: int | None = None,
    target_rows: Sequence[Mapping[str, Any]] | None = None,
) -> list[dict[str, Any]]:
    """Stream and normalize only source records referenced by the delta."""

    from cue_training.preprocessing.raw import process_dialoguestudio, process_lmsys, process_wildchat
    from cue_training.preprocessing.raw.conversation_utils import convert_chat_row, passes_filters

    if spec.kind == "dialogstudio":
        if not spec.config:
            raise ValueError("dialogstudio requires source_config")
        data = iter_dialogstudio_rows(spec, revision=revision)
    elif spec.kind in {"lmsys", "wildchat"}:
        from datasets import load_dataset

        load_kwargs: dict[str, Any] = {"split": spec.split, "streaming": True}
        if revision:
            load_kwargs["revision"] = revision
        data = load_dataset(spec.repo, **load_kwargs)
    else:
        raise ValueError(f"unsupported source kind: {spec.kind}")

    targets = list(target_rows or [])
    pending: dict[str, int] = {}
    for position, row in enumerate(targets):
        for key in (str(row["source_id"]), str(row.get("source_native_id") or "")):
            if key:
                pending.setdefault(key, position)
    unmatched = set(range(len(targets)))

    matched: list[dict[str, Any]] = []
    scanned = 0
    for index, row in enumerate(data):
        scanned += 1
        if max_records is not None and max_records > 0 and scanned > max_records:
            break
        if spec.kind == "dialogstudio":
            record = process_dialoguestudio.convert_dialog(
                row, str(spec.config), split=spec.split, index=index
            )
        else:
            config = process_lmsys.CONFIG if spec.kind == "lmsys" else process_wildchat.CONFIG
            # Only prefilter when scanning without targets; a requested conversation
            # must never be dropped by a filter the annotated corpus did not apply.
            if spec.kind == "wildchat" and not targets and not passes_filters(row, exclude_toxic=True):
                continue
            record = convert_chat_row(
                row,
                dataset_name=config.dataset_name,
                conversation_id=config.id_fn(row, index),
                split=spec.split,
            )
        if record is None:
            continue
        normalized = validate_record(record)
        keys = [
            str(normalized["id"]),
            source_native_id(
                str(normalized.get("dataset") or ""),
                str(normalized["id"]),
                normalized.get("metadata"),
            ),
        ]
        hit = next((pending[key] for key in keys if key in pending), None)
        if targets and hit is None:
            continue
        matched.append(normalized)
        if hit is not None:
            unmatched.discard(hit)
            target = targets[hit]
            for key in (str(target["source_id"]), str(target.get("source_native_id") or "")):
                pending.pop(key, None)
        if targets and not unmatched:
            break
    log(
        TAG,
        f"source scan rows={scanned} matched={len(matched)} "
        f"missing={len(unmatched)} repo={spec.repo}",
    )
    return matched


def _index_source_records(records: Iterable[Mapping[str, Any]]) -> tuple[
    dict[str, dict[str, Any]],
    dict[str, list[dict[str, Any]]],
]:
    by_id: dict[str, dict[str, Any]] = {}
    by_native: dict[str, list[dict[str, Any]]] = {}
    for record in records:
        rec = validate_record(dict(record))
        by_id[str(rec["id"])] = rec
        native = source_native_id(str(rec.get("dataset") or ""), str(rec["id"]), rec.get("metadata"))
        by_native.setdefault(native, []).append(rec)
    return by_id, by_native


def match_source_record(
    delta: Mapping[str, Any],
    *,
    by_id: Mapping[str, dict[str, Any]],
    by_native: Mapping[str, list[dict[str, Any]]],
) -> dict[str, Any]:
    """Locate a source conversation for one delta row by its source ids."""

    source_id = str(delta["source_id"])
    by_id_hit = by_id.get(source_id)
    if by_id_hit is not None:
        return by_id_hit
    candidates = by_native.get(str(delta.get("source_native_id") or ""), [])
    if candidates:
        return candidates[0]
    raise ValueError(f"no source match for dataset={delta.get('dataset')} source_id={source_id}")


def rehydrate_record(delta: Mapping[str, Any], source: Mapping[str, Any]) -> dict[str, Any]:
    """Join delta annotation onto a matched source conversation."""

    row = validate_delta_row(delta)
    src = validate_record(dict(source))
    provenance = {
        **sanitize_provenance(row.get("provenance")),
        "rehydrated": True,
        "source_repo": row["source_repo"],
        "source_config": row.get("source_config"),
        "source_revision": row.get("source_revision"),
        "annotation_version": row["annotation_version"],
    }
    metadata = {
        **dict(src.get("metadata") or {}),
        **dict(row.get("metadata") or {}),
        "rehydrated_from": {
            "source_repo": row["source_repo"],
            "source_config": row.get("source_config"),
            "source_id": row["source_id"],
        },
    }
    return validate_record(
        {
            "id": row["source_id"],
            "session_id": row["source_id"],
            "dataset": row["dataset"],
            "split": row["split"],
            "turns": src["turns"],
            "persona_manual": row["persona_manual"],
            "provenance": provenance,
            "metadata": metadata,
        }
    )


def rehydrate_delta_rows(
    delta_rows: Sequence[Mapping[str, Any]],
    *,
    source_loader=load_source_records,
    max_records_per_source: int | None = None,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Rehydrate many delta rows. Returns (ok_records, errors)."""

    grouped: dict[tuple[str, str | None, str, str | None], list[dict[str, Any]]] = {}
    for raw in delta_rows:
        row = validate_delta_row(raw)
        key = (
            str(row["source_repo"]),
            row.get("source_config"),
            str(row["source_split"]),
            row.get("source_revision"),
        )
        grouped.setdefault(key, []).append(row)

    out: list[dict[str, Any]] = []
    errors: list[dict[str, Any]] = []
    for (repo, config, split, revision), rows in grouped.items():
        dataset = rows[0]["dataset"]
        spec = SourceSpec(repo=repo, config=config, split=split, kind=resolve_source(dataset).kind)
        log(
            TAG,
            f"loading source repo={repo} config={config} split={split} "
            f"revision={revision} n_delta={len(rows)}",
        )
        try:
            source_records = source_loader(
                spec,
                revision=revision if revision else None,
                max_records=max_records_per_source,
                target_rows=rows,
            )
        except Exception as exc:  # noqa: BLE001
            for row in rows:
                errors.append({"source_id": row["source_id"], "dataset": row["dataset"], "error": str(exc)})
            continue
        by_id, by_native = _index_source_records(source_records)
        for row in rows:
            try:
                matched = match_source_record(row, by_id=by_id, by_native=by_native)
                out.append(rehydrate_record(row, matched))
            except Exception as exc:  # noqa: BLE001
                errors.append(
                    {"source_id": row["source_id"], "dataset": row["dataset"], "error": str(exc)}
                )
    return out, errors


def _first_difference(left: Sequence[Mapping[str, Any]], right: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Describe where two turn lists first diverge."""

    for index, (a, b) in enumerate(zip(left, right)):
        if a["role"] != b["role"]:
            return {"turn": index, "kind": "role", "annotated": a["role"], "source": b["role"]}
        if a["content"] != b["content"]:
            offset = next(
                (i for i, (x, y) in enumerate(zip(a["content"], b["content"])) if x != y),
                min(len(a["content"]), len(b["content"])),
            )
            return {
                "turn": index,
                "kind": "content",
                "char_offset": offset,
                "annotated": a["content"][offset : offset + 120],
                "source": b["content"][offset : offset + 120],
            }
    if len(left) != len(right):
        return {"kind": "turn_count", "annotated": len(left), "source": len(right)}
    return {"kind": "identical"}


def audit_annotated_records(
    annotated_root: str | Path,
    *,
    split: str = "train",
    datasets: Sequence[str] | None = None,
    sample_per_dataset: int = 25,
    source_loader=load_source_records,
    max_records_per_source: int | None = None,
) -> dict[str, Any]:
    """Diff local annotated turns against their upstream source, turn by turn."""

    by_dataset: dict[str, list[dict[str, Any]]] = {}
    allow = {d.strip() for d in datasets} if datasets else None
    for record in iter_annotated_records(annotated_root, split=split):
        dataset = canonical_dataset(str(record.get("dataset") or ""))
        if allow is not None and dataset not in allow:
            continue
        bucket = by_dataset.setdefault(dataset, [])
        if len(bucket) < sample_per_dataset:
            bucket.append(record)

    out: dict[str, Any] = {}
    for dataset, records in sorted(by_dataset.items()):
        rows = [build_delta_row(record) for record in records]
        spec = resolve_source(dataset, split=split)
        sources = source_loader(
            spec,
            revision=None,
            max_records=max_records_per_source,
            target_rows=rows,
        )
        by_id, by_native = _index_source_records(sources)
        found = identical = 0
        differences: list[dict[str, Any]] = []
        for record, row in zip(records, rows):
            try:
                match = match_source_record(row, by_id=by_id, by_native=by_native)
            except ValueError:
                continue
            found += 1
            difference = _first_difference(record["turns"], match["turns"])
            if difference["kind"] == "identical":
                identical += 1
                continue
            differences.append({"source_id": row["source_id"], **difference})
        out[dataset] = {
            "sampled": len(records),
            "found": found,
            "identical": identical,
            "differences": differences,
        }
        log(TAG, f"audit {dataset}: found {found}/{len(records)} identical={identical}")
    return out


def existing_record_ids(out_root: str | Path) -> dict[tuple[str, str], set[str]]:
    """Ids already written under `{dataset}/{split}.jsonl`, keyed by (dataset, split)."""

    root = Path(out_root)
    have: dict[tuple[str, str], set[str]] = {}
    if not root.is_dir():
        return have
    for path in sorted(root.glob("*/*.jsonl")):
        ids = have.setdefault((path.parent.name, path.stem), set())
        for record in iter_jsonl(path, validate=False):
            if isinstance(record, Mapping) and record.get("id"):
                ids.add(str(record["id"]))
    return have


def coverage_report(
    delta_rows: Sequence[Mapping[str, Any]],
    records: Sequence[Mapping[str, Any]],
    errors: Sequence[Mapping[str, Any]] = (),
) -> dict[str, Any]:
    """Per-dataset hydration coverage, so a partial pull can never pass silently."""

    requested: dict[str, set[str]] = {}
    for row in delta_rows:
        dataset = canonical_dataset(str(row.get("dataset") or ""))
        requested.setdefault(dataset, set()).add(str(row.get("source_id")))
    hydrated: dict[str, set[str]] = {}
    for record in records:
        dataset = canonical_dataset(str(record.get("dataset") or ""))
        hydrated.setdefault(dataset, set()).add(str(record.get("id")))

    sources: dict[str, Any] = {}
    for dataset, ids in sorted(requested.items()):
        got = hydrated.get(dataset, set())
        sources[dataset] = {
            "requested": len(ids),
            "hydrated": len(got),
            "coverage": (len(got) / len(ids)) if ids else 1.0,
            "missing_ids": sorted(ids - got),
        }
    total_requested = sum(item["requested"] for item in sources.values())
    total_hydrated = sum(item["hydrated"] for item in sources.values())
    return {
        "sources": sources,
        "requested": total_requested,
        "hydrated": total_hydrated,
        "coverage": (total_hydrated / total_requested) if total_requested else 1.0,
        "errors": len(errors),
    }


def write_rehydrated_dataset(
    records: Sequence[Mapping[str, Any]],
    out_root: str | Path,
    *,
    split: str | None = None,
    append: bool = False,
) -> list[Path]:
    """Write `{dataset}/{split}.jsonl` trees consumed by annotation/training.

    If ``split`` is set, all records are written under that filename. Otherwise each
    record's own ``split`` field is used (supports train + validation together).
    """

    out_root = Path(out_root)
    by_key: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for record in records:
        rec = validate_record(dict(record))
        if not rec.get("persona_manual"):
            raise ManualValidationError(f"rehydrated record {rec.get('id')} missing persona_manual")
        split_name = str(split or rec.get("split") or "train")
        by_key.setdefault((str(rec["dataset"]), split_name), []).append(rec)

    written: list[Path] = []
    for (dataset, split_name), items in sorted(by_key.items()):
        path = out_root / dataset / f"{split_name}.jsonl"
        write_jsonl_atomic(path, items, append=append)
        written.append(path)
        log(TAG, f"wrote {len(items)} -> {path}")
    if written:
        write_streaming_index(out_root, written, extra={"stage": "rehydrated"})
    return written


def _dataset_dict_from_splits(splits: Mapping[str, Sequence[Mapping[str, Any]]]):
    from datasets import Dataset, DatasetDict

    payload = {
        name: Dataset.from_list(list(items))
        for name, items in splits.items()
        if items
    }
    if not payload:
        raise ValueError("no non-empty splits to push")
    return DatasetDict(payload)


def push_delta_to_hub(
    rows: Sequence[Mapping[str, Any]],
    repo_id: str,
    *,
    private: bool = True,
    token: str | None = None,
    revision: str | None = None,
) -> str:
    """Push default + per-dataset HF configs, each with train/validation splits."""

    grouped = group_delta_by_dataset_split(rows)
    if not grouped:
        raise ValueError("no delta rows to push")

    # Default config: train/validation with all datasets mixed.
    default_splits: dict[str, list[dict[str, Any]]] = {}
    for split_map in grouped.values():
        for split_name, items in split_map.items():
            default_splits.setdefault(split_name, []).extend(items)
    _dataset_dict_from_splits(default_splits).push_to_hub(
        repo_id,
        config_name="default",
        private=private,
        token=token,
        revision=revision,
    )
    log(
        TAG,
        f"pushed config=default splits={{{', '.join(f'{k}: {len(v)}' for k, v in sorted(default_splits.items()))}}}",
    )

    for name, split_map in sorted(grouped.items()):
        _dataset_dict_from_splits(split_map).push_to_hub(
            repo_id,
            config_name=name,
            private=private,
            token=token,
            revision=revision,
        )
        counts = {split: len(items) for split, items in split_map.items()}
        log(TAG, f"pushed config={name} splits={counts}")
    return repo_id


def _decode_hub_row(item: Mapping[str, Any]) -> dict[str, Any]:
    row = dict(item)
    for key in ("persona_manual", "provenance", "metadata"):
        value = row.get(key)
        if isinstance(value, str):
            row[key] = json.loads(value)
    return validate_delta_row(row)


def load_delta_from_hub(
    repo_id: str,
    *,
    config_name: str | None = None,
    split: str | Sequence[str] | None = None,
    revision: str | None = None,
    token: str | None = None,
) -> list[dict[str, Any]]:
    """Load delta rows from Hub.

    - ``config_name=None`` loads every per-dataset config (skips the mixed ``default``).
    - ``split=None`` loads every available split (typically train + validation).
    """

    from datasets import get_dataset_config_names, get_dataset_split_names, load_dataset

    load_common: dict[str, Any] = {}
    if revision:
        load_common["revision"] = revision
    if token:
        load_common["token"] = token

    if config_name:
        configs = [config_name]
    else:
        # "default" and "all" are mixed views of the per-dataset configs; loading
        # them alongside would return every delta row twice.
        configs = [
            name
            for name in get_dataset_config_names(repo_id, **load_common)
            if name not in {"default", "all"}
        ]
        if not configs:
            configs = ["default"]

    if isinstance(split, str):
        wanted_splits: list[str] | None = [split]
    elif split is None:
        wanted_splits = None
    else:
        wanted_splits = [str(s) for s in split]

    rows: list[dict[str, Any]] = []
    for name in configs:
        available = list(get_dataset_split_names(repo_id, name, **load_common))
        use_splits = wanted_splits if wanted_splits is not None else available
        for split_name in use_splits:
            if split_name not in available:
                warn(TAG, f"missing split={split_name} config={name}; skipping")
                continue
            data = load_dataset(repo_id, name, split=split_name, **load_common)
            rows.extend(_decode_hub_row(item) for item in data)
    return rows
