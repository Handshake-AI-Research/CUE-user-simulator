"""Load CUE's preprocessed ``train.jsonl`` as plain conversations.

CUE's encoder corpus stores one record per user/session with ``turns`` carrying
raw ``user_text`` (the human turn) and ``system_text`` (the preceding assistant
context). We reconstruct alternating ``[{role, content}]`` transcripts from that
so the trainable baselines can reuse the same training corpus.
"""

from __future__ import annotations

import json
from collections import defaultdict
from collections.abc import Iterable, Iterator
from itertools import islice
from pathlib import Path
from typing import TypeVar

Turn = dict[str, str]

_T = TypeVar("_T")


def chunked(iterable: Iterable[_T], size: int) -> Iterator[list[_T]]:
    """Yield successive ``size``-length chunks from ``iterable`` (last may be short)."""

    it = iter(iterable)
    step = max(1, int(size))
    while batch := list(islice(it, step)):
        yield batch

# Default labeled training filename. The CUE train tree nests one
# ``<dataset>/train.jsonl`` (labeled) plus a large ``train.unlabeled.jsonl`` and
# other sidecars per dataset, so a bare ``*.jsonl`` glob would sweep in the
# unlabeled corpus. Mirror the encoder's ``--jsonl_names`` selection here.
DEFAULT_JSONL_NAMES: tuple[str, ...] = ("train.jsonl",)


def _jsonl_files(path: Path, jsonl_names: tuple[str, ...] = DEFAULT_JSONL_NAMES) -> list[Path]:
    if path.is_dir():
        names = set(jsonl_names)
        return sorted(p for p in path.rglob("*.jsonl") if p.name in names)
    return [path]


def _record_conversation(record: dict) -> list[Turn]:
    conversation: list[Turn] = []
    for turn in record.get("turns", []):
        system_text = turn.get("system_text")
        if system_text:
            conversation.append({"role": "assistant", "content": str(system_text)})
        user_text = turn.get("user_text")
        if user_text:
            conversation.append({"role": "user", "content": str(user_text)})
    return conversation


def iter_records(
    path: Path,
    limit: int | None = None,
    jsonl_names: tuple[str, ...] = DEFAULT_JSONL_NAMES,
) -> Iterator[dict]:
    count = 0
    for file in _jsonl_files(path, jsonl_names):
        with open(file, encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                yield json.loads(line)
                count += 1
                if limit is not None and count >= limit:
                    return


def iter_records_round_robin(
    path: Path,
    limit: int | None = None,
    jsonl_names: tuple[str, ...] = DEFAULT_JSONL_NAMES,
) -> Iterator[dict]:
    """Stream records by round-robin across the per-dataset JSONL files.

    The CUE train tree nests one ``<dataset>/train.jsonl`` per corpus. Reading them
    sequentially front-loads whichever dataset sorts first; round-robin interleaves
    one record from each file per cycle so a streamed/limited pass stays balanced
    across datasets. Files are read lazily one line at a time (no full materialization).
    """

    files = _jsonl_files(Path(path), jsonl_names)
    handles = [open(f, encoding="utf-8") for f in files]
    try:
        def _gen(handle):
            for line in handle:
                line = line.strip()
                if line:
                    yield json.loads(line)

        active = [_gen(h) for h in handles]
        count = 0
        while active:
            still: list[Iterator[dict]] = []
            for gen in active:
                record = next(gen, None)
                if record is None:
                    continue
                yield record
                count += 1
                if limit is not None and count >= limit:
                    return
                still.append(gen)
            active = still
    finally:
        for handle in handles:
            handle.close()


def iter_conversations(
    path: Path,
    limit: int | None = None,
    jsonl_names: tuple[str, ...] = DEFAULT_JSONL_NAMES,
) -> Iterator[list[Turn]]:
    """Yield one alternating transcript per session record."""

    for record in iter_records(path, limit=limit, jsonl_names=jsonl_names):
        conversation = _record_conversation(record)
        if conversation:
            yield conversation


def iter_user_turns(
    path: Path,
    limit: int | None = None,
    jsonl_names: tuple[str, ...] = DEFAULT_JSONL_NAMES,
) -> Iterator[str]:
    """Yield individual real user-turn texts (training targets)."""

    seen = 0
    for record in iter_records(path, jsonl_names=jsonl_names):
        for turn in record.get("turns", []):
            text = turn.get("user_text")
            if not text:
                continue
            yield str(text)
            seen += 1
            if limit is not None and seen >= limit:
                return


def iter_normalized_conversations(
    data_root: Path | None = None, limit: int | None = None
) -> Iterator[list[Turn]]:
    """Fallback corpus: real conversations from the vendored normalized data.

    Used by trainable baselines when no CUE ``train.jsonl`` is supplied, so that
    ``train`` is self-contained across the three evaluation domains.
    """

    from cue_training.baselines.common.domains import DOMAINS, normalized_path
    from cue_training.baselines.common.io import load_normalized, real_conversation

    count = 0
    for domain in DOMAINS.values():
        path = normalized_path(domain, data_root)
        if not path.exists():
            continue
        for record in load_normalized(path):
            conversation = real_conversation(record)
            if conversation:
                yield conversation
                count += 1
                if limit is not None and count >= limit:
                    return


def training_conversations(
    data_path: Path | None,
    data_root: Path | None = None,
    limit: int | None = None,
    jsonl_names: tuple[str, ...] = DEFAULT_JSONL_NAMES,
) -> Iterator[list[Turn]]:
    """Prefer CUE ``train.jsonl`` when given; else fall back to vendored data.

    When ``data_path`` is a directory, only files whose name is in ``jsonl_names``
    (default ``train.jsonl``) are read, matching the encoder's ``--jsonl_names`` so
    the large ``train.unlabeled.jsonl`` corpus is not swept in.
    """

    if data_path is not None and Path(data_path).exists():
        yield from iter_conversations(Path(data_path), limit=limit, jsonl_names=jsonl_names)
    else:
        yield from iter_normalized_conversations(data_root=data_root, limit=limit)


def group_by_user(
    path: Path,
    limit: int | None = None,
    jsonl_names: tuple[str, ...] = DEFAULT_JSONL_NAMES,
) -> dict[str, list[list[Turn]]]:
    """Group session transcripts by ``user_id`` for history-conditioned baselines."""

    grouped: dict[str, list[list[Turn]]] = defaultdict(list)
    for record in iter_records(path, limit=limit, jsonl_names=jsonl_names):
        conversation = _record_conversation(record)
        if conversation:
            grouped[str(record.get("user_id", "unknown"))].append(conversation)
    return dict(grouped)
