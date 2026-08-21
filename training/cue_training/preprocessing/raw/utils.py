"""Raw input and dataset-wise output helpers."""

from __future__ import annotations

import json
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any

from cue_training.data.streaming import write_jsonl_atomic, write_streaming_index


def load_records(path: str | Path) -> Any:
    path = Path(path)
    if path.suffix == ".jsonl":
        with path.open(encoding="utf-8") as handle:
            return [json.loads(line) for line in handle if line.strip()]
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def write_dataset_split(
    records: Iterable[Mapping[str, Any]],
    out_dir: str | Path,
    dataset: str,
    split: str,
) -> Path:
    root = Path(out_dir)
    path = root / dataset / f"{split}.jsonl"
    write_jsonl_atomic(path, (dict(record) for record in records))
    files = sorted(root.glob("*/*.jsonl"))
    write_streaming_index(root, files, extra={"stage": "raw"})
    return path
