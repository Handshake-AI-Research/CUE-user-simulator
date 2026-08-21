"""Offline Wegmann session-style targets for joint training relational distillation."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.distributed as dist
import torch.nn.functional as F

from cue_training.data.streaming import discover_jsonl_files, iter_jsonl
from cue_training.utils.log import log

TAG = "wegmann_style"
DEFAULT_SIDECAR = "wegmann_style.jsonl"


def _user_turn_texts(turns: list[dict[str, Any]]) -> list[str]:
    return [str(t.get("content") or "").strip() for t in turns if t.get("role") == "user"]


def encode_session_style(
    turns: list[dict[str, Any]],
    encoder: Any,
) -> np.ndarray | None:
    """Per-turn Wegmann encode, L2-normalize, mean-pool, renormalize."""

    texts = [t for t in _user_turn_texts(turns) if t]
    if not texts:
        return None
    vecs = encoder.encode(texts)
    if vecs.size == 0:
        return None
    norms = np.linalg.norm(vecs, axis=1, keepdims=True)
    vecs = vecs / np.maximum(norms, 1e-8)
    pooled = vecs.mean(axis=0)
    pooled = pooled / max(float(np.linalg.norm(pooled)), 1e-8)
    return pooled.astype(np.float32)


def encode_session_styles(
    sessions: list[list[dict[str, Any]]],
    encoder: Any,
) -> list[np.ndarray | None]:
    """Encode user turns from multiple sessions in one cross-session model pass."""

    all_texts: list[str] = []
    spans: list[tuple[int, int]] = []
    for turns in sessions:
        texts = [text for text in _user_turn_texts(turns) if text]
        start = len(all_texts)
        all_texts.extend(texts)
        spans.append((start, len(all_texts)))
    if not all_texts:
        return [None] * len(sessions)

    vectors = encoder.encode(all_texts)
    vectors = vectors / np.maximum(np.linalg.norm(vectors, axis=1, keepdims=True), 1e-8)
    styles: list[np.ndarray | None] = []
    for start, end in spans:
        if start == end:
            styles.append(None)
            continue
        pooled = vectors[start:end].mean(axis=0)
        pooled = pooled / max(float(np.linalg.norm(pooled)), 1e-8)
        styles.append(pooled.astype(np.float32))
    return styles


def _load_style_encoder(model_name: str, batch_size: int) -> Any:
    from cue_training.evaluation.metrics.metrics.style_encoder import get_encoder

    return get_encoder(model_name, batch_size=batch_size)


def _session_shard(session_id: str, num_shards: int) -> int:
    """Stable shard assignment via hashlib so shards stay balanced across datasets."""

    import hashlib

    digest = hashlib.md5(session_id.encode("utf-8")).hexdigest()
    return int(digest, 16) % max(1, num_shards)


def build_wegmann_sidecar(
    data_root: str | Path,
    output_path: str | Path,
    *,
    split: str = "train",
    model_name: str = "AnnaWegmann/Style-Embedding",
    batch_size: int = 32,
    session_batch_size: int = 256,
    max_records: int = 0,
    shard_index: int = 0,
    num_shards: int = 1,
) -> Path:
    """Precompute session-level style vectors for all records in split.

    With ``num_shards > 1``, only sessions assigned to ``shard_index`` are written.
    Run one process per GPU, then :func:`merge_wegmann_shards`.
    """

    data_root = Path(data_root)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if not 0 <= shard_index < max(1, num_shards):
        raise ValueError(f"shard_index={shard_index} out of range for num_shards={num_shards}")
    encoder = _load_style_encoder(model_name, batch_size)
    tmp = output_path.with_suffix(output_path.suffix + ".tmp")
    n_written = 0
    n_seen = 0
    pending_ids: list[str] = []
    pending_sessions: list[list[dict[str, Any]]] = []

    def flush(handle: Any) -> None:
        nonlocal n_written
        if not pending_ids:
            return
        styles = encode_session_styles(pending_sessions, encoder)
        for sid, style in zip(pending_ids, styles, strict=True):
            if style is None:
                continue
            handle.write(
                json.dumps(
                    {"session_id": sid, "style_embedding": style.tolist()},
                    ensure_ascii=False,
                )
                + "\n"
            )
            n_written += 1
        pending_ids.clear()
        pending_sessions.clear()

    with tmp.open("w", encoding="utf-8") as handle:
        for path in discover_jsonl_files(data_root, split=split):
            for record in iter_jsonl(path, validate=True):
                if max_records and n_seen >= max_records:
                    break
                sid = str(record.get("id") or record.get("session_id") or "")
                if not sid:
                    continue
                n_seen += 1
                if num_shards > 1 and _session_shard(sid, num_shards) != shard_index:
                    continue
                pending_ids.append(sid)
                pending_sessions.append(record.get("turns") or [])
                if len(pending_ids) >= max(1, session_batch_size):
                    flush(handle)
            if max_records and n_seen >= max_records:
                break
        flush(handle)
    os.replace(tmp, output_path)
    log(
        TAG,
        f"wrote {n_written} style vectors "
        f"(shard {shard_index}/{num_shards}, seen={n_seen}) -> {output_path}",
    )
    return output_path


def merge_wegmann_shards(
    shard_paths: list[str | Path],
    output_path: str | Path,
) -> Path:
    """Concatenate shard JSONLs, keeping the first embedding per session_id."""

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = output_path.with_suffix(output_path.suffix + ".tmp")
    seen: set[str] = set()
    n_written = 0
    with tmp.open("w", encoding="utf-8") as out:
        for shard in shard_paths:
            path = Path(shard)
            if not path.is_file():
                raise FileNotFoundError(f"missing Wegmann shard: {path}")
            with path.open(encoding="utf-8") as handle:
                for line in handle:
                    if not line.strip():
                        continue
                    row = json.loads(line)
                    sid = str(row.get("session_id") or "")
                    if not sid or sid in seen:
                        continue
                    seen.add(sid)
                    out.write(json.dumps(row, ensure_ascii=False) + "\n")
                    n_written += 1
    os.replace(tmp, output_path)
    log(TAG, f"merged {len(shard_paths)} shards -> {n_written} vectors at {output_path}")
    return output_path


def load_wegmann_sidecar(path: str | Path) -> dict[str, np.ndarray]:
    path = Path(path)
    if not path.is_file():
        return {}
    out: dict[str, np.ndarray] = {}
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            sid = str(row.get("session_id") or "")
            emb = row.get("style_embedding")
            if sid and isinstance(emb, list) and emb:
                out[sid] = np.asarray(emb, dtype=np.float32)
    return out


def sidecar_path(data_root: str | Path, split: str = "train") -> Path:
    return Path(data_root) / split / DEFAULT_SIDECAR


def batch_style_overlap(
    session_ids: list[str],
    style_map: dict[str, np.ndarray],
    device: torch.device,
) -> torch.Tensor | None:
    """Pairwise Wegmann cosine matrix [local_B, global_B] for relational distillation."""

    vecs: list[np.ndarray] = []
    for sid in session_ids:
        vector = style_map.get(str(sid))
        if vector is None:
            return None
        vecs.append(vector)
    local = torch.tensor(np.stack(vecs), dtype=torch.float32, device=device)
    local = F.normalize(local, dim=-1)
    global_vecs = local
    if dist.is_available() and dist.is_initialized():
        from torch.distributed.nn.functional import all_gather

        global_vecs = torch.cat(all_gather(local), dim=0)
    return local @ global_vecs.T
