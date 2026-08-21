"""Embedding loading helpers shared by evaluation harnesses."""

import json
from collections import defaultdict
from collections.abc import Mapping

import numpy as np


def read_embedding_rows(path: str) -> list[dict]:
    text = open(path, encoding="utf-8").read()
    try:
        rows = json.loads(text)
    except json.JSONDecodeError:
        rows = [json.loads(line) for line in text.splitlines() if line.strip()]
    if isinstance(rows, dict):
        rows = [rows] if "trajectory_embedding" in rows else rows.get(
            "records", rows.get("data", [])
        )
    return list(rows)


def load_embeddings(path: str) -> tuple[dict[str, list[float]], dict[str, list[float]]]:
    """Load exported vectors at their native encoder scale.

    Vectors must NOT be L2-normalized: they are fed straight back into the decoder as the
    bottleneck, whose LayerNorm puts it at norm ~sqrt(bottleneck_dim). Rescaling to unit norm
    shrinks the conditioning ~32x and the decoder emits a near-unconditional manual.
    """

    rows = read_embedding_rows(path)
    by_key: dict[str, list[float]] = {}
    by_domain: dict[str, list[np.ndarray]] = defaultdict(list)
    for row in rows:
        emb = row.get("trajectory_embedding")
        if emb is None:
            continue
        vector = np.asarray(emb, dtype=np.float32)
        metadata = row.get("metadata") if isinstance(row.get("metadata"), Mapping) else {}
        domain = str(metadata.get("domain") or row.get("dataset") or "")
        if domain:
            by_domain[domain].append(vector)
        for key in (
            row.get("session_id"),
            row.get("user_id"),
            metadata.get("episode_id"),
            metadata.get("task_id"),
            metadata.get("instance_id"),
        ):
            if key is not None:
                by_key[str(key)] = vector.tolist()
    means = {
        domain: np.stack(vectors).mean(axis=0).tolist()
        for domain, vectors in by_domain.items()
        if vectors
    }
    return by_key, means


def load_raw_embeddings(path: str) -> dict[str, list[float]]:
    """Load unnormalized vectors keyed like ``load_embeddings`` (sampler conditioning)."""

    by_key: dict[str, list[float]] = {}
    for row in read_embedding_rows(path):
        emb = row.get("trajectory_embedding")
        if emb is None:
            continue
        vector = np.asarray(emb, dtype=np.float32).tolist()
        metadata = row.get("metadata") if isinstance(row.get("metadata"), Mapping) else {}
        for key in (
            row.get("session_id"),
            row.get("user_id"),
            metadata.get("episode_id"),
            metadata.get("task_id"),
            metadata.get("instance_id"),
        ):
            if key is not None:
                by_key[str(key)] = vector
    return by_key
