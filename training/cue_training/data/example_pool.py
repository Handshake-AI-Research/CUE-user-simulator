"""CUE-keyed train example pool and retrieval helpers."""

from __future__ import annotations

import json
import os
import random
import shutil
from dataclasses import dataclass
from datetime import timedelta
from pathlib import Path
from typing import Any

import numpy as np
import torch
from tqdm.auto import tqdm

from cue_training.data.schema import filter_joint_training_target, validate_manual
from cue_training.data.streaming import discover_jsonl_files, iter_jsonl
from cue_training.infer.export import load_checkpoint
from cue_training.sampler.faiss_index import EmbeddingIndex
from cue_training.utils.log import log

TAG = "example_pool"
POOL_DIRNAME = "example_pool"
META_NAME = "meta.jsonl"
INDEX_NAME = "index.faiss"
MANIFEST_NAME = "manifest.json"
_POOL_CACHE: dict[str, ExamplePool] = {}


@dataclass(frozen=True)
class PoolExample:
    text: str
    kind: str  # sim_contrast | human_contrast


@dataclass
class ExamplePool:
    session_ids: list[str]
    cue_vectors: np.ndarray
    examples: list[list[PoolExample]]
    index: EmbeddingIndex
    checkpoint_fingerprint: str = ""

    @property
    def dim(self) -> int:
        return int(self.cue_vectors.shape[1])

    def save(self, directory: str | Path) -> Path:
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        np.save(directory / "cue_vectors.npy", self.cue_vectors.astype(np.float32))
        self.index.save(directory / INDEX_NAME)
        with (directory / META_NAME).open("w", encoding="utf-8") as handle:
            for sid, exs in zip(self.session_ids, self.examples, strict=True):
                handle.write(
                    json.dumps(
                        {
                            "session_id": sid,
                            "examples": [{"text": e.text, "kind": e.kind} for e in exs],
                        },
                        ensure_ascii=False,
                    )
                    + "\n"
                )
        manifest = {
            "n_sessions": len(self.session_ids),
            "dim": self.dim,
            "checkpoint_fingerprint": self.checkpoint_fingerprint,
        }
        (directory / MANIFEST_NAME).write_text(
            json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8"
        )
        return directory

    @classmethod
    def load(cls, directory: str | Path) -> ExamplePool:
        directory = Path(directory)
        vectors = np.load(directory / "cue_vectors.npy").astype(np.float32)
        dim = int(vectors.shape[1])
        index = EmbeddingIndex.load(directory / INDEX_NAME, dim)
        session_ids: list[str] = []
        examples: list[list[PoolExample]] = []
        with (directory / META_NAME).open(encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                row = json.loads(line)
                session_ids.append(str(row["session_id"]))
                examples.append(
                    [
                        PoolExample(text=str(e["text"]), kind=str(e["kind"]))
                        for e in (row.get("examples") or [])
                    ]
                )
        fingerprint = ""
        manifest_path = directory / MANIFEST_NAME
        if manifest_path.is_file():
            fingerprint = str(json.loads(manifest_path.read_text()).get("checkpoint_fingerprint") or "")
        return cls(session_ids, vectors, examples, index, fingerprint)

    def retrieve(
        self,
        cue: list[float] | np.ndarray,
        *,
        k_sessions: int = 8,
        n_general: int = 2,
        n_specific: int = 2,
        exclude_session_id: str | None = None,
    ) -> list[str]:
        query = np.asarray(cue, dtype=np.float32).reshape(1, -1)
        _, idxs = self.index.search(query, min(k_sessions, max(1, self.ntotal)))
        general: list[str] = []
        specific: list[str] = []
        seen: set[str] = set()
        for idx in idxs[0]:
            if idx < 0:
                continue
            i = int(idx)
            if exclude_session_id and self.session_ids[i] == exclude_session_id:
                continue
            for ex in self.examples[i]:
                key = ex.text.casefold()
                if key in seen:
                    continue
                seen.add(key)
                if ex.kind == "sim_contrast" and len(general) < n_general:
                    general.append(ex.text)
                elif ex.kind != "sim_contrast" and len(specific) < n_specific:
                    specific.append(ex.text)
        out = general + specific
        if len(out) < n_general + n_specific:
            for idx in idxs[0]:
                if idx < 0:
                    continue
                i = int(idx)
                for ex in self.examples[i]:
                    key = ex.text.casefold()
                    if key in seen:
                        continue
                    seen.add(key)
                    out.append(ex.text)
                    if len(out) >= n_general + n_specific:
                        break
                if len(out) >= n_general + n_specific:
                    break
        return out[: n_general + n_specific]

    @property
    def ntotal(self) -> int:
        return len(self.session_ids)


def examples_from_manual(manual: dict[str, Any]) -> list[PoolExample]:
    return _examples_from_manual(manual)


def _examples_from_manual(manual: dict[str, Any]) -> list[PoolExample]:
    filtered = filter_joint_training_target(manual)
    validated = validate_manual(filtered)
    out: list[PoolExample] = []
    for cmd in validated["commands"]:
        kind = str(cmd.get("kind") or "human_contrast")
        for ex in cmd.get("examples") or []:
            text = str(ex).strip()
            if text:
                out.append(PoolExample(text=text, kind=kind))
    return out


def _session_shard(session_id: str, num_shards: int) -> int:
    """Stable shard assignment via hashlib so shards stay balanced across datasets."""

    import hashlib

    digest = hashlib.md5(session_id.encode("utf-8")).hexdigest()
    return int(digest, 16) % max(1, num_shards)


def build_example_pool(
    checkpoint: str | Path,
    data_root: str | Path,
    output_dir: str | Path,
    *,
    split: str = "train",
    batch_size: int = 256,
    device: str = "cpu",
    max_records: int = 0,
    shard_index: int = 0,
    num_shards: int = 1,
) -> ExamplePool:
    """Encode the split and save a CUE-keyed example pool.

    With ``num_shards > 1`` only sessions assigned to ``shard_index`` are pooled, so one
    process per GPU can run in parallel; combine the results with
    :func:`merge_example_pools`.
    """

    data_root = Path(data_root)
    checkpoint = Path(checkpoint)
    if not 0 <= shard_index < max(1, num_shards):
        raise ValueError(f"shard_index={shard_index} out of range for num_shards={num_shards}")
    model = load_checkpoint(checkpoint, device=device)
    model.eval()
    session_ids: list[str] = []
    cue_vectors: list[np.ndarray] = []
    examples: list[list[PoolExample]] = []
    batch_records: list[dict[str, Any]] = []

    def _flush() -> None:
        nonlocal batch_records
        if not batch_records:
            return
        with torch.no_grad():
            embs = model.encode([r["turns"] for r in batch_records])
        for record, emb in zip(batch_records, embs, strict=True):
            exs = _examples_from_manual(record["persona_manual"])
            if not exs:
                continue
            session_ids.append(record["session_id"])
            cue_vectors.append(emb.detach().float().cpu().numpy())
            examples.append(exs)
        batch_records = []

    n_seen = 0
    with tqdm(
        total=(max_records // max(1, num_shards)) or None,
        desc=f"build example pool (shard {shard_index}/{num_shards})",
        unit="session",
    ) as bar:
        for path in discover_jsonl_files(data_root, split=split):
            for record in iter_jsonl(path, validate=True):
                if not record.get("persona_manual"):
                    continue
                sid = str(record.get("id") or record.get("session_id") or "")
                if not sid:
                    continue
                n_seen += 1
                if num_shards > 1 and _session_shard(sid, num_shards) != shard_index:
                    continue
                batch_records.append(
                    {
                        "session_id": sid,
                        "turns": record["turns"],
                        "persona_manual": record["persona_manual"],
                    }
                )
                bar.update()
                if len(batch_records) >= batch_size:
                    _flush()
                    bar.set_postfix(pooled=len(session_ids))
                if max_records and n_seen >= max_records:
                    break
            if max_records and n_seen >= max_records:
                break
    _flush()
    if not session_ids:
        raise ValueError(f"no pool entries built from {data_root}/{split}")
    stacked = np.stack(cue_vectors, axis=0).astype(np.float32)
    index = EmbeddingIndex(stacked.shape[1])
    index.add(stacked)
    pool = ExamplePool(session_ids, stacked, examples, index, str(checkpoint.resolve()))
    pool.save(output_dir)
    log(TAG, f"built pool n={len(session_ids)} -> {output_dir}")
    return pool


def build_example_pool_distributed(
    checkpoint: str | Path,
    data_root: str | Path,
    output_dir: str | Path,
    *,
    split: str = "train",
    batch_size: int = 256,
    device: str = "cpu",
    max_records: int = 0,
) -> ExamplePool | None:
    """Build the pool under ``torchrun``: one shard per rank, merged by rank 0.

    Without torchrun (``WORLD_SIZE`` unset or 1) this is just :func:`build_example_pool`.
    Under torchrun each rank encodes on its own ``LOCAL_RANK`` GPU, so ``device`` is
    ignored, and shards are written next to ``output_dir`` then removed after the merge.
    Returns the merged pool on rank 0 and ``None`` on other ranks.
    """

    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    output_dir = Path(output_dir)
    if world_size <= 1:
        return build_example_pool(
            checkpoint, data_root, output_dir, split=split,
            batch_size=batch_size, device=device, max_records=max_records,
        )

    import torch.distributed as dist

    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    rank_device = f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu"
    shards = [output_dir.with_name(f"{output_dir.name}.shard{i}") for i in range(world_size)]
    if not dist.is_initialized():
        # gloo: the only collective is a barrier, so this needs no GPU backend. The long
        # timeout covers uneven shards (one rank can finish well before another).
        dist.init_process_group(backend="gloo", timeout=timedelta(hours=6))
    build_example_pool(
        checkpoint, data_root, shards[rank], split=split, batch_size=batch_size,
        device=rank_device, max_records=max_records,
        shard_index=rank, num_shards=world_size,
    )
    dist.barrier()
    pool = None
    if rank == 0:
        pool = merge_example_pools(shards, output_dir)
        for path in shards:
            shutil.rmtree(path, ignore_errors=True)
    dist.barrier()
    dist.destroy_process_group()
    return pool


def merge_example_pools(
    shard_dirs: list[str | Path],
    output_dir: str | Path,
) -> ExamplePool:
    """Concatenate sharded pools into one, keeping the first entry per session_id.

    The FAISS index is rebuilt rather than merged, so the result is identical to a
    single-process build over the same sessions.
    """

    session_ids: list[str] = []
    vectors: list[np.ndarray] = []
    examples: list[list[PoolExample]] = []
    fingerprints: set[str] = set()
    seen: set[str] = set()
    for shard in shard_dirs:
        path = Path(shard)
        if not (path / META_NAME).is_file():
            raise FileNotFoundError(f"missing example-pool shard: {path}")
        pool = ExamplePool.load(path)
        fingerprints.add(pool.checkpoint_fingerprint)
        for i, sid in enumerate(pool.session_ids):
            if sid in seen:
                continue
            seen.add(sid)
            session_ids.append(sid)
            vectors.append(pool.cue_vectors[i])
            examples.append(pool.examples[i])
    if not session_ids:
        raise ValueError(f"no pool entries across {len(shard_dirs)} shards")
    # Vectors from different encoders are not comparable, so refuse to merge them.
    if len(fingerprints) > 1:
        raise ValueError(f"shards built from different checkpoints: {sorted(fingerprints)}")
    stacked = np.stack(vectors, axis=0).astype(np.float32)
    index = EmbeddingIndex(stacked.shape[1])
    index.add(stacked)
    pool = ExamplePool(session_ids, stacked, examples, index, fingerprints.pop())
    pool.save(output_dir)
    log(TAG, f"merged {len(shard_dirs)} shards -> n={len(session_ids)} at {output_dir}")
    return pool


def pool_dir_for_checkpoint(checkpoint: str | Path) -> Path:
    return Path(checkpoint) / POOL_DIRNAME


def resolve_pool_checkpoint(decoder_dir: str | Path) -> Path | None:
    path = Path(decoder_dir)
    if (path / POOL_DIRNAME / META_NAME).is_file():
        return path
    refine_cfg = path / "refinement_config.json"
    if refine_cfg.is_file():
        payload = json.loads(refine_cfg.read_text(encoding="utf-8"))
        joint_ckpt = payload.get("joint_training_checkpoint")
        if joint_ckpt:
            candidate = Path(str(joint_ckpt))
            if (candidate / POOL_DIRNAME / META_NAME).is_file():
                return candidate
    return None


def load_pool_for_checkpoint(checkpoint: str | Path) -> ExamplePool | None:
    directory = pool_dir_for_checkpoint(checkpoint)
    if not (directory / META_NAME).is_file():
        return None
    return ExamplePool.load(directory)


def get_pool_for_decoder_dir(decoder_dir: str | Path) -> ExamplePool | None:
    checkpoint = resolve_pool_checkpoint(decoder_dir)
    if checkpoint is None:
        return None
    key = str(checkpoint.resolve())
    if key not in _POOL_CACHE:
        _POOL_CACHE[key] = ExamplePool.load(checkpoint / POOL_DIRNAME)
    return _POOL_CACHE[key]


def inject_examples_into_dual_render(
    general_text: str,
    specific_text: str,
    cue_embedding: list[float] | np.ndarray,
    pool: ExamplePool,
    *,
    exclude_session_id: str | None = None,
    n_general: int = 2,
    n_specific: int = 2,
    k_sessions: int = 8,
) -> str:
    from cue_training.data.schema import parse_manual
    from cue_training.refinement.simulator import render_dual_manual, render_manual

    general = parse_manual(general_text) or {"commands": [], "examples": []}
    specific = parse_manual(specific_text) or {"commands": [], "examples": []}
    general_out, specific_out, _ = augment_manual_block(
        general,
        specific,
        pool=pool,
        cue_embedding=cue_embedding,
        exclude_session_id=exclude_session_id,
        n_general=n_general,
        n_specific=n_specific,
        k_sessions=k_sessions,
    )
    return render_dual_manual(general_out, specific_out) or render_manual(
        {
            "commands": list(general_out.get("commands") or [])
            + list(specific_out.get("commands") or []),
            "examples": list(specific_out.get("examples") or []),
        }
    )


def augment_manual_block(
    general: dict[str, Any],
    specific: dict[str, Any],
    *,
    pool: ExamplePool,
    cue_embedding: list[float] | np.ndarray,
    exclude_session_id: str | None = None,
    n_general: int = 2,
    n_specific: int = 2,
    k_sessions: int = 8,
) -> tuple[dict[str, Any], dict[str, Any], list[str]]:
    retrieved = pool.retrieve(
        cue_embedding,
        k_sessions=k_sessions,
        n_general=n_general,
        n_specific=n_specific,
        exclude_session_id=exclude_session_id,
    )
    general_out = {**general, "examples": []}
    specific_out = {**specific, "examples": retrieved}
    return general_out, specific_out, retrieved


def _jaccard_tokens(a: str, b: str) -> float:
    ta = frozenset(a.casefold().split())
    tb = frozenset(b.casefold().split())
    if not ta and not tb:
        return 1.0
    return len(ta & tb) / max(1, len(ta | tb))


def eval_validation_retrieval(
    *,
    gallery_ids: list[str],
    gallery_vectors: np.ndarray,
    gallery_examples: list[list[PoolExample]],
    query_ids: list[str],
    query_vectors: np.ndarray,
    query_target_examples: list[list[str]],
    k_sessions: int = 8,
    n_general: int = 2,
    n_specific: int = 2,
) -> dict[str, float]:
    if not gallery_ids or not query_ids:
        return {}
    index = EmbeddingIndex(gallery_vectors.shape[1])
    index.add(gallery_vectors.astype(np.float32))
    coverage_hits = 0
    backfill_hits = 0
    diversity_scores: list[float] = []
    overlap_scores: list[float] = []
    shuffled_overlap_scores: list[float] = []
    rng = random.Random(0)

    for qid, qvec, targets in zip(query_ids, query_vectors, query_target_examples, strict=True):
        _, idxs = index.search(qvec.reshape(1, -1), min(k_sessions, len(gallery_ids)))
        general: list[str] = []
        specific: list[str] = []
        seen: set[str] = set()
        for idx in idxs[0]:
            if idx < 0:
                continue
            gi = int(idx)
            if gallery_ids[gi] == qid:
                continue
            for ex in gallery_examples[gi]:
                key = ex.text.casefold()
                if key in seen:
                    continue
                seen.add(key)
                if ex.kind == "sim_contrast" and len(general) < n_general:
                    general.append(ex.text)
                elif ex.kind != "sim_contrast" and len(specific) < n_specific:
                    specific.append(ex.text)
        retrieved = general + specific
        if len(general) >= n_general and len(specific) >= n_specific:
            coverage_hits += 1
        if len(retrieved) < n_general + n_specific:
            backfill_hits += 1
        if len(retrieved) >= 2:
            pairs = [
                _jaccard_tokens(retrieved[i], retrieved[j])
                for i in range(len(retrieved))
                for j in range(i + 1, len(retrieved))
            ]
            diversity_scores.append(1.0 - float(np.mean(pairs)))
        if targets and retrieved:
            best = max(_jaccard_tokens(t, r) for t in targets for r in retrieved)
            overlap_scores.append(best)
            shuffled_gi = rng.choice([i for i in range(len(gallery_ids)) if gallery_ids[i] != qid])
            shuffled_ex = [e.text for e in gallery_examples[shuffled_gi]]
            if shuffled_ex:
                shuffled_best = max(_jaccard_tokens(t, r) for t in targets for r in shuffled_ex)
                shuffled_overlap_scores.append(shuffled_best)

    n = max(1, len(query_ids))
    return {
        "retrieval_coverage": coverage_hits / n,
        "retrieval_kind_backfill_rate": backfill_hits / n,
        "retrieval_diversity": float(np.mean(diversity_scores)) if diversity_scores else 0.0,
        "retrieval_overlap": float(np.mean(overlap_scores)) if overlap_scores else 0.0,
        "retrieval_shuffled_overlap": float(np.mean(shuffled_overlap_scores)) if shuffled_overlap_scores else 0.0,
        "retrieval_gap": (
            float(np.mean(overlap_scores)) - float(np.mean(shuffled_overlap_scores))
            if overlap_scores and shuffled_overlap_scores
            else 0.0
        ),
    }
