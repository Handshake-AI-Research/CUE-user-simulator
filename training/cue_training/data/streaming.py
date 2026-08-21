"""Dataset-wise streaming JSONL IO and round-robin IterableDataset."""

from __future__ import annotations

import json
import os
import random
from collections.abc import Iterable, Iterator
from pathlib import Path
from typing import Any

from torch.utils.data import IterableDataset

from cue_training.data.schema import validate_record
from cue_training.utils.log import warn

TAG = "stream"


def discover_jsonl_files(root: str | Path, *, split: str = "train") -> list[Path]:
    """Find `{dataset}/{split}.jsonl` (and same-dataset parts) under root."""

    root = Path(root)
    if root.is_file() and root.suffix == ".jsonl":
        return [root]
    files: list[Path] = []
    if not root.is_dir():
        return files
    for path in sorted(root.rglob("*.jsonl")):
        name = path.name
        if name.endswith(".partial.jsonl") or name.endswith(".error.jsonl"):
            continue
        if name == f"{split}.jsonl" or (
            path.parent.name == split and name.startswith("part-") and name.endswith(".jsonl")
        ):
            files.append(path)
    return files


def iter_jsonl(path: str | Path, *, validate: bool = True) -> Iterator[dict[str, Any]]:
    path = Path(path)
    with path.open("r", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                warn(TAG, f"skip corrupt JSON {path}:{line_no}")
                continue
            if validate:
                try:
                    record = validate_record(record)
                except ValueError as exc:
                    warn(TAG, f"skip invalid record {path}:{line_no}: {exc}")
                    continue
            yield record


def write_jsonl_atomic(
    path: str | Path,
    records: Iterable[dict[str, Any]],
    *,
    append: bool = False,
) -> int:
    """Write JSONL via temp file + rename (or append to .partial then rename)."""

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    # Always start the temp file fresh, so a leftover .tmp from an interrupted run
    # cannot be folded into the output.
    count = 0
    with tmp.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
            count += 1
    if append and path.exists():
        with path.open("a", encoding="utf-8") as out, tmp.open("r", encoding="utf-8") as src:
            out.write(src.read())
        tmp.unlink(missing_ok=True)
    else:
        os.replace(tmp, path)
    return count


def count_from_streaming_index(
    root: str | Path,
    files: list[Path],
) -> int | None:
    """Sum ``count`` for ``files`` from ``root/streaming_index.json``, or None if missing.

    Keys in the index are paths relative to ``root`` (as written by
    :func:`write_streaming_index`). Files outside ``root`` or absent from the index
    are skipped; if none of ``files`` resolve, returns None so callers keep an
    indeterminate tqdm total.
    """

    root = Path(root)
    index_path = root / "streaming_index.json"
    if not index_path.is_file() or not files:
        return None
    try:
        with index_path.open(encoding="utf-8") as handle:
            payload = json.load(handle)
    except (OSError, json.JSONDecodeError):
        return None
    entries = payload.get("files") or {}
    if not isinstance(entries, dict):
        return None
    total = 0
    matched = 0
    for path in files:
        try:
            rel = str(path.resolve().relative_to(root.resolve()))
        except ValueError:
            continue
        entry = entries.get(rel)
        if not isinstance(entry, dict) or "count" not in entry:
            continue
        total += int(entry["count"])
        matched += 1
    return total if matched else None


def write_streaming_index(
    root: str | Path,
    files: list[Path],
    *,
    schema_version: int = 1,
    extra: dict[str, Any] | None = None,
) -> Path:
    root = Path(root)
    entries: dict[str, Any] = {}
    for path in files:
        count = 0
        dataset = path.parent.name if path.parent != root else path.stem
        with path.open("r", encoding="utf-8") as handle:
            for line in handle:
                if line.strip():
                    count += 1
        rel = str(path.relative_to(root)) if path.is_relative_to(root) else str(path)
        entries[rel] = {
            "dataset": dataset,
            "count": count,
            "bytes": path.stat().st_size,
        }
    payload = {
        "schema_version": schema_version,
        "files": entries,
        **(extra or {}),
    }
    out = root / "streaming_index.json"
    tmp = out.with_suffix(".tmp")
    with tmp.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
    os.replace(tmp, out)
    return out


class RoundRobinStreamingDataset(IterableDataset):
    """Lazy round-robin over dataset-wise JSONL files with epoch caps."""

    def __init__(
        self,
        data_root: str | Path,
        *,
        split: str = "train",
        per_source_cap: int = 10_000,
        shuffle_buffer: int = 256,
        seed: int = 0,
        require_manual: bool = True,
    ) -> None:
        self.data_root = Path(data_root)
        self.split = split
        self.per_source_cap = max(0, per_source_cap)
        self.shuffle_buffer = max(0, shuffle_buffer)
        self.seed = seed
        self.require_manual = require_manual
        self.files = discover_jsonl_files(self.data_root, split=split)
        if not self.files:
            raise FileNotFoundError(f"No {split}.jsonl under {self.data_root}")
        self.epoch = 0
        self.rank = 0
        self.world_size = 1

    def configure(self, *, rank: int = 0, world_size: int = 1, epoch: int = 0) -> RoundRobinStreamingDataset:
        self.rank = rank
        self.world_size = max(1, world_size)
        self.epoch = epoch
        return self

    def set_epoch(self, epoch: int) -> None:
        self.epoch = epoch

    def state_dict(self) -> dict[str, Any]:
        return {
            "epoch": self.epoch,
            "rank": self.rank,
            "world_size": self.world_size,
            "seed": self.seed,
            "files": [str(p) for p in self.files],
        }

    def load_state_dict(self, state: dict[str, Any]) -> None:
        self.epoch = int(state.get("epoch", 0))
        self.rank = int(state.get("rank", 0))
        self.world_size = int(state.get("world_size", 1))

    def _file_iter(self, path: Path) -> Iterator[dict[str, Any]]:
        rng = random.Random(self.seed + self.epoch + hash(str(path)) % 10_000)
        buffer: list[dict[str, Any]] = []
        yielded = 0
        for index, record in enumerate(iter_jsonl(path, validate=True)):
            if self.require_manual and not record.get("persona_manual"):
                continue
            if (index % self.world_size) != self.rank:
                continue
            if self.per_source_cap and yielded >= self.per_source_cap:
                break
            if self.shuffle_buffer > 0:
                buffer.append(record)
                if len(buffer) >= self.shuffle_buffer:
                    pick = rng.randrange(len(buffer))
                    yielded += 1
                    yield buffer.pop(pick)
            else:
                yielded += 1
                yield record
        rng.shuffle(buffer)
        for record in buffer:
            if self.per_source_cap and yielded >= self.per_source_cap:
                break
            yielded += 1
            yield record

    def __iter__(self) -> Iterator[dict[str, Any]]:
        iterators = [self._file_iter(path) for path in self.files]
        while iterators:
            active: list[Iterator[dict[str, Any]]] = []
            for iterator in iterators:
                try:
                    yield next(iterator)
                    active.append(iterator)
                except StopIteration:
                    pass
            iterators = active
