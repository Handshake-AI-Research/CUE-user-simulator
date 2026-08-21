"""Build and cache the prior embedding bank + FAISS index the sampler trains on."""

from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass, field
from datetime import timedelta
from pathlib import Path
from typing import Any, Iterator

import numpy as np
import torch
from tqdm.auto import tqdm

from cue_training.data.schema import validate_record
from cue_training.data.streaming import count_from_streaming_index, iter_jsonl
from cue_training.infer.export import load_checkpoint
from cue_training.sampler.faiss_index import EmbeddingIndex, _l2_normalize
from cue_training.utils.config import storage_root
from cue_training.utils.log import log, warn

TAG = "sampler.bank"
META_NAME = "bank_meta.json"
REPORT_NAME = "bank_report.json"
DEFAULT_STORAGE_ROOT = storage_root()


@dataclass
class BankShard:
    name: str
    embeddings: np.ndarray  # memmap or ndarray [N, D]
    session_ids: list[str]
    sources: list[str]

    @property
    def n(self) -> int:
        return int(self.embeddings.shape[0])

    @property
    def dim(self) -> int:
        return int(self.embeddings.shape[1])


@dataclass
class EmbeddingBank:
    root: Path
    dim: int
    mu: np.ndarray
    sigma: np.ndarray
    layernorm_weight: np.ndarray | None
    layernorm_bias: np.ndarray | None
    checkpoint_hash: str
    prior_train: BankShard
    prior_dev: BankShard
    prior_index: EmbeddingIndex | None = None
    meta: dict[str, Any] | None = None
    report: dict[str, Any] | None = None

    def standardize(self, z: np.ndarray | torch.Tensor) -> np.ndarray | torch.Tensor:
        if torch.is_tensor(z):
            mu = torch.as_tensor(self.mu, device=z.device, dtype=z.dtype)
            sigma = torch.as_tensor(self.sigma, device=z.device, dtype=z.dtype)
            return (z - mu) / sigma
        return ((z - self.mu) / self.sigma).astype(np.float32)

    def unstandardize(self, z: np.ndarray | torch.Tensor) -> np.ndarray | torch.Tensor:
        if torch.is_tensor(z):
            mu = torch.as_tensor(self.mu, device=z.device, dtype=z.dtype)
            sigma = torch.as_tensor(self.sigma, device=z.device, dtype=z.dtype)
            return z * sigma + mu
        return (z * self.sigma + self.mu).astype(np.float32)


def discover_files_by_name(root: str | Path, filename: str) -> list[Path]:
    """Find ``**/filename`` under root, skipping .partial / .error sidecars."""

    root = Path(root)
    if root.is_file() and root.name == filename:
        return [root]
    if not root.is_dir():
        return []
    files: list[Path] = []
    for path in sorted(root.rglob(filename)):
        name = path.name
        if name.endswith(".partial.jsonl") or name.endswith(".error.jsonl"):
            continue
        if name == filename:
            files.append(path)
    return files


def checkpoint_content_hash(checkpoint: str | Path) -> str:
    checkpoint = Path(checkpoint)
    h = hashlib.sha256()
    for name in ("unified_config.json", "model.pt", "refinement_config.json", "decoder_state.pt"):
        path = checkpoint / name
        if not path.is_file():
            continue
        h.update(name.encode())
        with path.open("rb") as handle:
            while True:
                chunk = handle.read(1 << 20)
                if not chunk:
                    break
                h.update(chunk)
    return h.hexdigest()[:16]


def extract_bottleneck_layernorm(model: Any) -> tuple[np.ndarray | None, np.ndarray | None]:
    bn = getattr(getattr(model, "encoder", None), "bottleneck", None)
    if bn is None or not hasattr(bn, "__getitem__"):
        return None, None
    try:
        ln = bn[1]
        weight = ln.weight.detach().float().cpu().numpy()
        bias = ln.bias.detach().float().cpu().numpy()
        return weight, bias
    except Exception:  # noqa: BLE001
        return None, None


def _session_text_hash(turns: list[dict[str, Any]]) -> str:
    parts = []
    for turn in turns:
        parts.append(f"{turn.get('role', '')}:{turn.get('content', '')}")
    return hashlib.md5("\n".join(parts).encode("utf-8")).hexdigest()


def _is_degenerate(vec: np.ndarray, eps: float = 1e-6) -> bool:
    if not np.isfinite(vec).all():
        return True
    return float(np.linalg.norm(vec)) < eps


def _session_shard(session_id: str, num_shards: int) -> int:
    """Stable shard assignment so ranks see disjoint, balanced record subsets."""

    return int(hashlib.md5(session_id.encode("utf-8")).hexdigest(), 16) % max(1, num_shards)


def _iter_bank_records(
    files: list[Path],
    *,
    per_source_cap: int = 0,
    shard_index: int = 0,
    num_shards: int = 1,
) -> Iterator[dict[str, Any]]:
    per_source: dict[str, int] = {}
    # Shard first, then cap, so the cap counts only this rank's records; scale it down so
    # the union across ranks still honors the configured per-source total.
    if per_source_cap > 0 and num_shards > 1:
        per_source_cap = -(-per_source_cap // num_shards)
    for path in files:
        source = path.parent.name
        for record in iter_jsonl(path, validate=False):
            try:
                if "turns" not in record and "conversation" in record:
                    record = {**record, "turns": record["conversation"]}
                if "turns" not in record and "real_conversation" in record:
                    record = {**record, "turns": record["real_conversation"]}
                rec = validate_record(record)
            except ValueError:
                continue
            if num_shards > 1 and _session_shard(rec["session_id"], num_shards) != shard_index:
                continue
            if per_source_cap > 0:
                n = per_source.get(source, 0)
                if n >= per_source_cap:
                    continue
                per_source[source] = n + 1
            rec["_source"] = source
            rec["_text_hash"] = _session_text_hash(rec["turns"])
            yield rec


def _holdout_bucket(session_id: str, *, seed: int, holdout_frac: float) -> bool:
    """True when the session belongs in the holdout/dev split."""

    if holdout_frac <= 0:
        return False
    bucket = int(hashlib.md5(f"{seed}:{session_id}".encode()).hexdigest()[:8], 16) / 0xFFFFFFFF
    return bucket < holdout_frac


def _holdout_split(
    items: list[dict[str, Any]],
    *,
    holdout_frac: float,
    seed: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    if holdout_frac <= 0 or len(items) < 2:
        return items, []
    rng = np.random.default_rng(seed)
    train: list[dict[str, Any]] = []
    dev: list[dict[str, Any]] = []
    for item in items:
        if _holdout_bucket(item["session_id"], seed=seed, holdout_frac=holdout_frac):
            dev.append(item)
        else:
            train.append(item)
    if not dev and items:
        idx = int(rng.integers(0, len(items)))
        moved = items[idx]
        train = [x for i, x in enumerate(items) if i != idx]
        dev = [moved]
    return train, dev


@dataclass
class _StreamingShardWriter:
    """Append embeddings + lightweight meta while streaming records."""

    out_root: Path
    name: str
    dim: int
    n: int = 0
    stats: dict[str, int] = field(
        default_factory=lambda: {
            "degenerate": 0,
            "encoded": 0,
            "text_dup": 0,
            "near_dup_dropped": 0,
        }
    )
    _seen_hash: set[str] = field(default_factory=set, repr=False)
    _raw: Any = field(default=None, repr=False)
    _meta: Any = field(default=None, repr=False)

    def __post_init__(self) -> None:
        self.out_root.mkdir(parents=True, exist_ok=True)
        self._raw_path = self.out_root / f"{self.name}.raw.bin"
        self._meta_tmp = self.out_root / f"{self.name}.meta.jsonl.tmp"
        self._raw = self._raw_path.open("wb")
        self._meta = self._meta_tmp.open("w", encoding="utf-8")

    def add_encoded(self, records: list[dict[str, Any]], embeddings: np.ndarray) -> int:
        kept = 0
        for rec, vec in zip(records, embeddings, strict=True):
            self.stats["encoded"] += 1
            if _is_degenerate(vec):
                self.stats["degenerate"] += 1
                continue
            th = str(rec.get("_text_hash") or "")
            if th and th in self._seen_hash:
                self.stats["text_dup"] += 1
                continue
            if th:
                self._seen_hash.add(th)
            self._raw.write(np.asarray(vec, dtype=np.float32).tobytes(order="C"))
            self._meta.write(
                json.dumps(
                    {
                        "row": self.n,
                        "session_id": rec["session_id"],
                        "source": rec.get("_source") or rec.get("dataset") or "unknown",
                        "text_hash": th or None,
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )
            self.n += 1
            kept += 1
        return kept

    def close_handles(self) -> None:
        if self._raw is not None:
            self._raw.close()
            self._raw = None
        if self._meta is not None:
            self._meta.close()
            self._meta = None

    def finalize(
        self,
        *,
        near_dup_cosine: float = 1.0,
    ) -> tuple[np.ndarray, Path, dict[str, int]]:
        self.close_handles()
        emb_path = self.out_root / f"{self.name}.npy"
        meta_path = self.out_root / f"{self.name}.meta.jsonl"
        emb = _raw_bin_to_npy(self._raw_path, emb_path, self.n, self.dim)
        os.replace(self._meta_tmp, meta_path)
        if emb.shape[0] and near_dup_cosine < 1.0:
            emb, dropped = _rewrite_without_near_dups(
                emb_path,
                meta_path,
                cosine_threshold=near_dup_cosine,
            )
            self.stats["near_dup_dropped"] = dropped
            self.n = int(emb.shape[0])
            if dropped:
                log(
                    TAG,
                    f"{self.name}: dropped {dropped} near-duplicates "
                    f"(cos>={near_dup_cosine})",
                )
        return emb, meta_path, dict(self.stats)


def _raw_bin_to_npy(raw_path: Path, npy_path: Path, n: int, dim: int) -> np.ndarray:
    if n <= 0:
        if raw_path.exists():
            raw_path.unlink()
        empty = np.zeros((0, dim), dtype=np.float32)
        np.save(npy_path, empty)
        return np.load(npy_path, mmap_mode="r")
    expected = n * dim * 4
    actual = raw_path.stat().st_size
    if actual != expected:
        raise RuntimeError(
            f"raw embedding size mismatch for {raw_path}: {actual} bytes != {expected}"
        )
    src = np.memmap(raw_path, dtype=np.float32, mode="r", shape=(n, dim))
    out = np.lib.format.open_memmap(npy_path, mode="w+", dtype=np.float32, shape=(n, dim))
    chunk = 8192
    for start in range(0, n, chunk):
        end = min(n, start + chunk)
        out[start:end] = src[start:end]
    out.flush()
    del src
    raw_path.unlink(missing_ok=True)
    return np.load(npy_path, mmap_mode="r")


def _rewrite_without_near_dups(
    emb_path: Path,
    meta_path: Path,
    *,
    cosine_threshold: float,
) -> tuple[np.ndarray, int]:
    emb = np.load(emb_path, mmap_mode="r")
    keep = _near_duplicate_mask(emb, cosine_threshold=cosine_threshold)
    dropped = int((~keep).sum())
    if dropped == 0:
        return emb, 0
    n_keep = int(keep.sum())
    dim = int(emb.shape[1])
    tmp_emb = emb_path.with_suffix(".npy.filtered")
    out = np.lib.format.open_memmap(tmp_emb, mode="w+", dtype=np.float32, shape=(n_keep, dim))
    write_i = 0
    chunk = 8192
    for start in range(0, emb.shape[0], chunk):
        end = min(emb.shape[0], start + chunk)
        mask = keep[start:end]
        if not mask.any():
            continue
        rows = emb[start:end][mask]
        out[write_i : write_i + rows.shape[0]] = rows
        write_i += rows.shape[0]
    out.flush()
    del out
    os.replace(tmp_emb, emb_path)

    def _filter_jsonl(path: Path) -> None:
        tmp = path.with_suffix(path.suffix + ".filtered")
        with path.open(encoding="utf-8") as src, tmp.open("w", encoding="utf-8") as dst:
            kept_row = 0
            for index, line in enumerate(src):
                if not line.strip() or not keep[index]:
                    continue
                row = json.loads(line)
                row["row"] = kept_row
                dst.write(json.dumps(row, ensure_ascii=False) + "\n")
                kept_row += 1
        os.replace(tmp, path)

    _filter_jsonl(meta_path)
    return np.load(emb_path, mmap_mode="r"), dropped


@torch.no_grad()
def _stream_encode_records(
    model: Any,
    records: Iterator[dict[str, Any]],
    writers: dict[str, _StreamingShardWriter],
    *,
    route: Any,
    batch_size: int,
    desc: str,
    max_records: int = 0,
    total: int | None = None,
) -> None:
    """Encode a record stream into one or more shard writers without materializing it.

    ``total`` is only for the progress bar (typically from ``streaming_index.json``).
    ``max_records`` still hard-caps how many sessions are encoded.
    """

    buffers: dict[str, list[dict[str, Any]]] = {name: [] for name in writers}
    seen = 0
    bar = tqdm(total=(max_records or total or None), desc=desc, unit="session")
    try:
        for record in records:
            if max_records and seen >= max_records:
                break
            name = route(record)
            if name not in writers:
                continue
            buffers[name].append(record)
            seen += 1
            bar.update()
            if len(buffers[name]) >= batch_size:
                batch = buffers[name]
                buffers[name] = []
                emb = model.encode([r["turns"] for r in batch])
                writers[name].add_encoded(batch, emb.detach().float().cpu().numpy())
                bar.set_postfix(
                    **{key: writer.n for key, writer in writers.items()},
                    refresh=False,
                )
        for name, batch in buffers.items():
            if not batch:
                continue
            emb = model.encode([r["turns"] for r in batch])
            writers[name].add_encoded(batch, emb.detach().float().cpu().numpy())
        bar.set_postfix(**{key: writer.n for key, writer in writers.items()})
    finally:
        bar.close()


@torch.no_grad()
def _encode_records(
    model: Any,
    records: list[dict[str, Any]],
    *,
    batch_size: int,
    device: str,
) -> tuple[np.ndarray, list[dict[str, Any]], dict[str, int]]:
    """Compatibility helper for small in-memory lists (tests / tiny shards)."""

    del device
    kept: list[dict[str, Any]] = []
    vectors: list[np.ndarray] = []
    stats = {"degenerate": 0, "encoded": 0, "text_dup": 0}
    seen_hash: set[str] = set()
    for start in range(0, len(records), batch_size):
        batch = records[start : start + batch_size]
        emb = model.encode([r["turns"] for r in batch])
        emb_np = emb.detach().float().cpu().numpy()
        for rec, vec in zip(batch, emb_np, strict=True):
            stats["encoded"] += 1
            if _is_degenerate(vec):
                stats["degenerate"] += 1
                continue
            th = rec.get("_text_hash") or ""
            if th and th in seen_hash:
                stats["text_dup"] += 1
                continue
            if th:
                seen_hash.add(th)
            kept.append(rec)
            vectors.append(vec.astype(np.float32))
    if not vectors:
        return np.zeros((0, 1024), dtype=np.float32), [], stats
    return np.stack(vectors, axis=0), kept, stats


def _near_duplicate_mask(
    embeddings: np.ndarray,
    *,
    cosine_threshold: float,
    chunk: int = 4096,
) -> np.ndarray:
    """Keep lowest-index occurrence; drop later near-duplicates (cosine >= threshold)."""

    n = embeddings.shape[0]
    keep = np.ones(n, dtype=bool)
    if n == 0 or cosine_threshold >= 1.0:
        return keep
    norms = _l2_normalize(embeddings)
    index = EmbeddingIndex(embeddings.shape[1])
    index.add(norms)
    for start in range(0, n, chunk):
        end = min(n, start + chunk)
        sims, idxs = index.search(norms[start:end], 2)
        for local, global_i in enumerate(range(start, end)):
            # Prefer the second neighbor when the first is self.
            cand = int(idxs[local, 0])
            sim = float(sims[local, 0])
            if cand == global_i and sims.shape[1] > 1:
                cand = int(idxs[local, 1])
                sim = float(sims[local, 1])
            if cand < 0 or cand == global_i:
                continue
            if sim >= cosine_threshold and cand < global_i:
                keep[global_i] = False
    return keep


def _write_memmap(path: Path, array: np.ndarray) -> np.ndarray:
    path.parent.mkdir(parents=True, exist_ok=True)
    mm = np.lib.format.open_memmap(path, mode="w+", dtype=np.float32, shape=array.shape)
    mm[:] = array
    mm.flush()
    return np.load(path, mmap_mode="r")


def _write_meta_rows(path: Path, records: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as handle:
        for i, rec in enumerate(records):
            row = {
                "row": i,
                "session_id": rec["session_id"],
                "source": rec.get("_source") or rec.get("dataset") or "unknown",
                "text_hash": rec.get("_text_hash"),
            }
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    os.replace(tmp, path)


def _load_meta_ids(path: Path) -> tuple[list[str], list[str]]:
    session_ids: list[str] = []
    sources: list[str] = []
    if not path.is_file():
        return session_ids, sources
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            session_ids.append(str(row["session_id"]))
            sources.append(str(row.get("source") or "unknown"))
    return session_ids, sources


def _shard_stats(embeddings: np.ndarray, session_ids: list[str], sources: list[str]) -> dict[str, Any]:
    if embeddings.shape[0] == 0:
        return {"n": 0}
    mean = embeddings.mean(axis=0)
    std = embeddings.std(axis=0)
    # NN distance histogram on a sample
    sample_n = min(2048, embeddings.shape[0])
    rng = np.random.default_rng(0)
    idx = rng.choice(embeddings.shape[0], size=sample_n, replace=False)
    sample = embeddings[idx]
    index = EmbeddingIndex(embeddings.shape[1])
    index.add(embeddings)
    sims, nn_idx = index.search(sample, 2)
    # second neighbor (exclude self)
    nn_sims = []
    for i in range(sample_n):
        # if first is self-ish, take second
        nn_sims.append(float(sims[i, 1] if sims.shape[1] > 1 else sims[i, 0]))
    nn_sims_arr = np.asarray(nn_sims, dtype=np.float32)
    from collections import Counter

    return {
        "n": int(embeddings.shape[0]),
        "dim": int(embeddings.shape[1]),
        "per_dim_mean_mean": float(mean.mean()),
        "per_dim_mean_std": float(mean.std()),
        "per_dim_std_mean": float(std.mean()),
        "per_dim_std_min": float(std.min()),
        "per_dim_std_max": float(std.max()),
        "nn_cosine_p50": float(np.median(nn_sims_arr)),
        "nn_cosine_p10": float(np.quantile(nn_sims_arr, 0.1)),
        "nn_cosine_p90": float(np.quantile(nn_sims_arr, 0.9)),
        "unique_session_ids": len(set(session_ids)),
        "source_counts": dict(Counter(sources)),
    }


def _sliced_wasserstein(a: np.ndarray, b: np.ndarray, *, projections: int = 64, seed: int = 0) -> float:
    if a.shape[0] == 0 or b.shape[0] == 0:
        return float("nan")
    rng = np.random.default_rng(seed)
    dim = a.shape[1]
    dirs = rng.normal(size=(projections, dim)).astype(np.float32)
    dirs /= np.linalg.norm(dirs, axis=1, keepdims=True) + 1e-8
    pa = a @ dirs.T
    pb = b @ dirs.T
    pa.sort(axis=0)
    pb.sort(axis=0)
    n = min(pa.shape[0], pb.shape[0])
    # resample to equal length via quantile interp
    qa = np.quantile(pa, np.linspace(0, 1, n), axis=0)
    qb = np.quantile(pb, np.linspace(0, 1, n), axis=0)
    return float(np.mean(np.abs(qa - qb)))


def _rff_mmd(a: np.ndarray, b: np.ndarray, *, features: int = 256, seed: int = 0) -> float:
    if a.shape[0] == 0 or b.shape[0] == 0:
        return float("nan")
    rng = np.random.default_rng(seed)
    dim = a.shape[1]
    w = rng.normal(size=(features, dim)).astype(np.float32)
    b_phase = rng.uniform(0, 2 * np.pi, size=(features,)).astype(np.float32)

    def featurize(x: np.ndarray) -> np.ndarray:
        return np.cos(x @ w.T + b_phase) * np.sqrt(2.0 / features)

    fa = featurize(a).mean(axis=0)
    fb = featurize(b).mean(axis=0)
    return float(np.sum((fa - fb) ** 2))


def build_bank(
    cfg: dict[str, Any],
    *,
    shard_index: int = 0,
    num_shards: int = 1,
    finalize: bool = True,
) -> EmbeddingBank | None:
    """Encode the prior corpus into a cached bank and index.

    With ``num_shards > 1`` this encodes only the sessions hashing to ``shard_index``.
    ``finalize=False`` stops after the per-shard embedding/meta files are written
    and returns ``None``, leaving mu/sigma, the FAISS index, and the report to
    :func:`merge_bank_shards` on rank 0 -- those are global quantities that cannot be
    computed from one shard.
    """

    out_root = Path(cfg.get("bank_dir") or cfg.get("output") or DEFAULT_STORAGE_ROOT / "models" / "cue" / "retrieval_bank")
    out_root.mkdir(parents=True, exist_ok=True)
    checkpoint = cfg["joint_training_checkpoint"]
    device = str(cfg.get("device", "cuda" if torch.cuda.is_available() else "cpu"))
    batch_size = int(cfg.get("encode_batch_size", 16))
    holdout_frac = float(cfg.get("prior_holdout_frac", 0.02))
    near_dup_cos = float(cfg.get("near_dup_cosine", 0.995))
    seed = int(cfg.get("seed", 0))
    ckpt_hash = checkpoint_content_hash(checkpoint)

    prior_root = Path(cfg.get("prior_data_root", DEFAULT_STORAGE_ROOT / "data"))
    prior_files = discover_files_by_name(
        prior_root,
        cfg.get("prior_filename", "train.jsonl"),
    )
    if not prior_files:
        raise FileNotFoundError(
            f"No prior files matching {cfg.get('prior_filename')} under {cfg.get('prior_data_root')}"
        )

    # The record cap is global, so each shard takes its share.
    max_prior = -(-int(cfg.get("max_prior_records", 0)) // max(1, num_shards))
    indexed = count_from_streaming_index(prior_root, prior_files)
    # Session-hash sharding is roughly even; ceil so the bar does not hit 100% early.
    encode_total = (
        None
        if indexed is None
        else -(-indexed // max(1, num_shards))
    )

    log(TAG, f"loading encoder from {checkpoint}")
    model = load_checkpoint(checkpoint, device=device)
    model.eval()
    ln_w, ln_b = extract_bottleneck_layernorm(model)
    dim = int(getattr(getattr(model, "encoder", None), "bottleneck_dim", 1024) or 1024)

    encode_stats: dict[str, Any] = {}
    shard_paths: dict[str, tuple[np.ndarray, Path]] = {}

    log(TAG, f"streaming prior from {len(prior_files)} files"
        + (f" (~{encode_total} sessions this shard)" if encode_total else ""))
    prior_writers = {
        "prior_train": _StreamingShardWriter(out_root, "prior_train", dim),
        "prior_dev": _StreamingShardWriter(out_root, "prior_dev", dim),
    }
    try:
        _stream_encode_records(
            model,
            _iter_bank_records(
                prior_files,
                per_source_cap=0,
                shard_index=shard_index,
                num_shards=num_shards,
            ),
            prior_writers,
            route=lambda rec: (
                "prior_dev"
                if _holdout_bucket(
                    rec["session_id"], seed=seed, holdout_frac=holdout_frac
                )
                else "prior_train"
            ),
            batch_size=batch_size,
            desc="encode prior",
            max_records=max_prior,
            total=encode_total,
        )
    except Exception:
        for writer in prior_writers.values():
            writer.close_handles()
        raise
    for name, writer in prior_writers.items():
        emb, meta_path, stats = writer.finalize(near_dup_cosine=near_dup_cos)
        encode_stats[name] = stats
        shard_paths[name] = (emb, meta_path)
        log(TAG, f"{name}: kept {emb.shape[0]} embeddings")

    if not finalize:
        # Encoder-derived, identical on every rank; the merge reads it back rather than
        # reloading the checkpoint just to recover the bottleneck LayerNorm affine.
        if ln_w is not None:
            np.save(out_root / "layernorm_weight.npy", ln_w)
            np.save(out_root / "layernorm_bias.npy", ln_b)
        log(TAG, f"shard {shard_index}/{num_shards} encoded -> {out_root} (not finalized)")
        return None

    return _finalize_bank(
        out_root,
        shard_paths=shard_paths,
        encode_stats=encode_stats,
        cfg=cfg,
        dim=dim,
        checkpoint=checkpoint,
        ckpt_hash=ckpt_hash,
        ln_w=ln_w,
        ln_b=ln_b,
        prior_files=prior_files,
        near_dup_cos=near_dup_cos,
        seed=seed,
    )


def _finalize_bank(
    out_root: Path,
    *,
    shard_paths: dict[str, tuple[np.ndarray, Path]],
    encode_stats: dict[str, Any],
    cfg: dict[str, Any],
    dim: int,
    checkpoint: Any,
    ckpt_hash: str,
    ln_w: np.ndarray | None,
    ln_b: np.ndarray | None,
    prior_files: list[Path],
    near_dup_cos: float,
    seed: int,
) -> EmbeddingBank:
    """Compute the global bank artifacts: mu/sigma, FAISS index, report, meta."""

    prior_emb = shard_paths["prior_train"][0]
    if prior_emb.shape[0] == 0:
        raise RuntimeError("prior_train bank is empty after filtering")
    mu = np.asarray(prior_emb, dtype=np.float32).mean(axis=0).astype(np.float32)
    sigma = np.asarray(prior_emb, dtype=np.float32).std(axis=0).astype(np.float32)
    sigma = np.maximum(sigma, 1e-4)

    bank_shards: dict[str, BankShard] = {}
    for name, (emb, meta_path) in shard_paths.items():
        session_ids, sources = _load_meta_ids(meta_path)
        bank_shards[name] = BankShard(
            name=name,
            embeddings=emb,
            session_ids=session_ids,
            sources=sources,
        )

    # FAISS index (train only)
    prior_index = EmbeddingIndex(dim)
    prior_index.add(np.asarray(bank_shards["prior_train"].embeddings))
    prior_index.save(out_root / "prior.faiss")

    # Baselines: train vs dev distributional distance
    pt = np.asarray(bank_shards["prior_train"].embeddings)
    pd = np.asarray(bank_shards["prior_dev"].embeddings)
    if pd.shape[0] == 0:
        # fallback subsample of train as faux-dev for baseline bookkeeping
        rng = np.random.default_rng(seed)
        take = min(1024, pt.shape[0] // 5) or min(64, pt.shape[0])
        pd = pt[rng.choice(pt.shape[0], size=take, replace=False)]
    pt_s = (pt - mu) / sigma
    pd_s = (pd - mu) / sigma
    # subsample for metrics
    rng = np.random.default_rng(seed)
    def _sub(x: np.ndarray, n: int = 4096) -> np.ndarray:
        if x.shape[0] <= n:
            return x
        return x[rng.choice(x.shape[0], size=n, replace=False)]

    sw_base = _sliced_wasserstein(_sub(pt_s), _sub(pd_s), seed=seed)
    mmd_base = _rff_mmd(_sub(pt_s), _sub(pd_s), seed=seed)

    report = {
        "checkpoint": str(checkpoint),
        "checkpoint_hash": ckpt_hash,
        "encode_stats": encode_stats,
        "shards": {
            name: _shard_stats(np.asarray(sh.embeddings), sh.session_ids, sh.sources)
            for name, sh in bank_shards.items()
        },
        "baselines": {
            "real_train_vs_dev_sliced_wasserstein": sw_base,
            "real_train_vs_dev_rff_mmd": mmd_base,
            "real_embedding_parse_rate": None,  # filled optionally by eval decode pass
        },
        "near_dup_cosine": near_dup_cos,
        "index_backend": prior_index.backend,
    }
    with (out_root / REPORT_NAME).open("w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2, sort_keys=True)

    meta = {
        "dim": dim,
        "checkpoint": str(checkpoint),
        "checkpoint_hash": ckpt_hash,
        "mu": mu.tolist(),
        "sigma": sigma.tolist(),
        "layernorm_weight": None if ln_w is None else ln_w.tolist(),
        "layernorm_bias": None if ln_b is None else ln_b.tolist(),
        "counts": {name: sh.n for name, sh in bank_shards.items()},
        "prior_files": [str(p) for p in prior_files],
        "config": {
            k: cfg[k]
            for k in (
                "prior_data_root",
                "prior_filename",
                "near_dup_cosine",
                "prior_holdout_frac",
                "seed",
            )
            if k in cfg
        },
    }
    tmp_meta = out_root / (META_NAME + ".tmp")
    with tmp_meta.open("w", encoding="utf-8") as handle:
        json.dump(meta, handle, indent=2, sort_keys=True)
    os.replace(tmp_meta, out_root / META_NAME)
    np.save(out_root / "mu.npy", mu)
    np.save(out_root / "sigma.npy", sigma)
    if ln_w is not None:
        np.save(out_root / "layernorm_weight.npy", ln_w)
        np.save(out_root / "layernorm_bias.npy", ln_b)

    log(TAG, f"bank ready at {out_root} (report={REPORT_NAME})")
    return load_bank(out_root)


_SHARD_NAMES = ("prior_train", "prior_dev")


def _concat_shard(
    out_root: Path, shard_dirs: list[Path], name: str, dim: int
) -> tuple[np.ndarray, Path, int]:
    """Concatenate one named shard across rank directories, dropping cross-rank text dups.

    Each rank deduplicated by text hash only within its own stream, so the same session
    text reaching two ranks would survive; filter on the merged meta here. Rows are
    renumbered so ``row`` stays the index into the merged embedding matrix.
    """

    emb_path = out_root / f"{name}.npy"
    meta_path = out_root / f"{name}.meta.jsonl"

    keep: list[tuple[Path, int, dict[str, Any]]] = []
    seen: set[str] = set()
    dropped = 0
    for directory in shard_dirs:
        shard_meta = directory / f"{name}.meta.jsonl"
        if not shard_meta.is_file():
            continue
        with shard_meta.open(encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                row = json.loads(line)
                text_hash = row.get("text_hash")
                if text_hash and text_hash in seen:
                    dropped += 1
                    continue
                if text_hash:
                    seen.add(text_hash)
                keep.append((directory, int(row["row"]), row))

    total = len(keep)
    out = np.lib.format.open_memmap(emb_path, mode="w+", dtype=np.float32, shape=(total, dim))
    sources = {d: np.load(d / f"{name}.npy", mmap_mode="r") for d in shard_dirs if (d / f"{name}.npy").is_file()}
    with meta_path.open("w", encoding="utf-8") as meta_out:
        for new_row, (directory, old_row, row) in enumerate(keep):
            out[new_row] = sources[directory][old_row]
            meta_out.write(json.dumps({**row, "row": new_row}, ensure_ascii=False) + "\n")
    out.flush()
    del out, sources
    return np.load(emb_path, mmap_mode="r"), meta_path, dropped


def merge_bank_shards(cfg: dict[str, Any], shard_dirs: list[str | Path]) -> EmbeddingBank:
    """Merge per-rank shard directories into one bank, then compute global artifacts."""

    out_root = Path(cfg.get("bank_dir") or cfg.get("output") or DEFAULT_STORAGE_ROOT / "models" / "cue" / "retrieval_bank")
    out_root.mkdir(parents=True, exist_ok=True)
    dirs = [Path(d) for d in shard_dirs]
    checkpoint = cfg["joint_training_checkpoint"]
    ckpt_hash = checkpoint_content_hash(checkpoint)
    near_dup_cos = float(cfg.get("near_dup_cosine", 0.995))
    seed = int(cfg.get("seed", 0))

    dim = 0
    ln_w = ln_b = None
    for directory in dirs:
        candidate = directory / "prior_train.npy"
        if candidate.is_file():
            dim = int(np.load(candidate, mmap_mode="r").shape[1])
            break
    if dim <= 0:
        raise RuntimeError(f"no prior_train.npy found in any shard dir: {dirs}")
    for stem, attr in (("layernorm_weight", "ln_w"), ("layernorm_bias", "ln_b")):
        for directory in dirs:
            path = directory / f"{stem}.npy"
            if path.is_file():
                if attr == "ln_w":
                    ln_w = np.load(path)
                else:
                    ln_b = np.load(path)
                break

    shard_paths: dict[str, tuple[np.ndarray, Path]] = {}
    encode_stats: dict[str, Any] = {}
    for name in _SHARD_NAMES:
        emb, meta_path, dropped = _concat_shard(out_root, dirs, name, dim)
        near_dropped = 0
        if emb.shape[0] and near_dup_cos < 1.0:
            # Each rank only saw near-dups inside its own shard; run the pass again over
            # the merged matrix so cross-rank pairs are caught too.
            emb, near_dropped = _rewrite_without_near_dups(
                out_root / f"{name}.npy", meta_path, cosine_threshold=near_dup_cos
            )
        shard_paths[name] = (emb, meta_path)
        encode_stats[name] = {
            "encoded": int(emb.shape[0]),
            "cross_shard_text_dup": dropped,
            "cross_shard_near_dup_dropped": near_dropped,
        }
        log(
            TAG,
            f"merged {name}: {emb.shape[0]} embeddings "
            f"({dropped} cross-shard text dups, {near_dropped} cross-shard near-dups)",
        )

    prior_files = discover_files_by_name(
        cfg.get("prior_data_root", DEFAULT_STORAGE_ROOT / "data"),
        cfg.get("prior_filename", "train.jsonl"),
    )
    return _finalize_bank(
        out_root,
        shard_paths=shard_paths,
        encode_stats=encode_stats,
        cfg=cfg,
        dim=dim,
        checkpoint=checkpoint,
        ckpt_hash=ckpt_hash,
        ln_w=ln_w,
        ln_b=ln_b,
        prior_files=prior_files,
        near_dup_cos=near_dup_cos,
        seed=seed,
    )


def build_bank_distributed(cfg: dict[str, Any]) -> EmbeddingBank | None:
    """Build the bank under ``torchrun``: one encode shard per rank, merged by rank 0.

    Without torchrun this is just :func:`build_bank`. Under torchrun each rank encodes on
    its own ``LOCAL_RANK`` GPU into ``<bank_dir>/.rank{k}``, then rank 0 concatenates the
    shards and computes mu/sigma, the FAISS index, and the report -- global quantities
    no single shard can produce. Returns the bank on rank 0 and ``None`` elsewhere.
    """

    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    if world_size <= 1:
        return build_bank(cfg)

    import shutil

    import torch.distributed as dist

    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    out_root = Path(cfg.get("bank_dir") or cfg.get("output") or DEFAULT_STORAGE_ROOT / "models" / "cue" / "retrieval_bank")
    shard_dirs = [out_root / f".rank{i}" for i in range(world_size)]
    if not dist.is_initialized():
        # gloo: the only collective is a barrier, so no GPU backend is needed. Long
        # timeout because shards finish at different times on uneven data.
        dist.init_process_group(backend="gloo", timeout=timedelta(hours=12))
    shard_cfg = {
        **cfg,
        "bank_dir": str(shard_dirs[rank]),
        "device": f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu",
    }
    build_bank(shard_cfg, shard_index=rank, num_shards=world_size, finalize=False)
    dist.barrier()
    bank = None
    if rank == 0:
        bank = merge_bank_shards(cfg, shard_dirs)
        for directory in shard_dirs:
            shutil.rmtree(directory, ignore_errors=True)
    dist.barrier()
    dist.destroy_process_group()
    return bank


def load_bank(bank_dir: str | Path) -> EmbeddingBank:
    root = Path(bank_dir)
    with (root / META_NAME).open(encoding="utf-8") as handle:
        meta = json.load(handle)
    dim = int(meta["dim"])
    mu = np.load(root / "mu.npy").astype(np.float32)
    sigma = np.load(root / "sigma.npy").astype(np.float32)
    ln_w = np.load(root / "layernorm_weight.npy").astype(np.float32) if (root / "layernorm_weight.npy").is_file() else None
    ln_b = np.load(root / "layernorm_bias.npy").astype(np.float32) if (root / "layernorm_bias.npy").is_file() else None

    def _load_shard(name: str) -> BankShard:
        emb_path = root / f"{name}.npy"
        emb = np.load(emb_path, mmap_mode="r") if emb_path.is_file() else np.zeros((0, dim), dtype=np.float32)
        meta_path = root / f"{name}.meta.jsonl"
        session_ids: list[str] = []
        sources: list[str] = []
        if meta_path.is_file():
            with meta_path.open(encoding="utf-8") as handle:
                for line in handle:
                    if not line.strip():
                        continue
                    row = json.loads(line)
                    session_ids.append(str(row["session_id"]))
                    sources.append(str(row.get("source") or "unknown"))
        return BankShard(
            name=name,
            embeddings=emb,
            session_ids=session_ids,
            sources=sources,
        )

    report = None
    report_path = root / REPORT_NAME
    if report_path.is_file():
        with report_path.open(encoding="utf-8") as handle:
            report = json.load(handle)

    def _load_index(name: str, shard_name: str) -> EmbeddingIndex:
        path = root / f"{name}.faiss"
        try:
            if path.is_file() or Path(str(path) + ".npy").is_file():
                return EmbeddingIndex.load(path, dim)
        except Exception as exc:  # noqa: BLE001
            warn(TAG, f"could not load {name} index: {exc}")
        index = EmbeddingIndex(dim)
        sh = _load_shard(shard_name)
        if sh.n:
            index.add(np.asarray(sh.embeddings))
        return index

    prior_index = _load_index("prior", "prior_train")

    return EmbeddingBank(
        root=root,
        dim=dim,
        mu=mu,
        sigma=sigma,
        layernorm_weight=ln_w,
        layernorm_bias=ln_b,
        checkpoint_hash=str(meta.get("checkpoint_hash") or ""),
        prior_train=_load_shard("prior_train"),
        prior_dev=_load_shard("prior_dev"),
        prior_index=prior_index,
        meta=meta,
        report=report,
    )

