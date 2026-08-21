"""CUE-keyed example pool: retrieval and injection into decoded manuals.

The pool is large (one embedding plus example texts per training session), so it ships
as a separate Hugging Face **dataset** rather than inside the model repo. Pools built by
cue_training's ``cue-train build-example-pool`` are also loadable straight from disk.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from cue_hf.faiss_index import EmbeddingIndex
from cue_hf.log import log

TAG = "example_pool"
POOL_DIRNAME = "example_pool"
META_NAME = "meta.jsonl"
INDEX_NAME = "index.faiss"
MANIFEST_NAME = "manifest.json"
VECTORS_NAME = "cue_vectors.npy"


@dataclass(frozen=True)
class PoolExample:
    text: str
    kind: str  # sim_contrast | human_contrast | style


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

    @property
    def ntotal(self) -> int:
        return len(self.session_ids)

    def retrieve(
        self,
        cue: list[float] | np.ndarray,
        *,
        k_sessions: int = 8,
        n_general: int = 2,
        n_specific: int = 6,
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

    @classmethod
    def from_rows(
        cls,
        session_ids: list[str],
        cue_vectors: np.ndarray,
        examples: list[list[PoolExample]],
        *,
        checkpoint_fingerprint: str = "",
    ) -> ExamplePool:
        vectors = np.asarray(cue_vectors, dtype=np.float32)
        if vectors.ndim != 2:
            raise ValueError(f"cue_vectors must be [N, D], got {vectors.shape}")
        index = EmbeddingIndex(int(vectors.shape[1]))
        index.add(vectors)
        return cls(
            list(session_ids),
            vectors,
            list(examples),
            index,
            checkpoint_fingerprint,
        )

    @classmethod
    def from_directory(cls, directory: str | Path) -> ExamplePool:
        """Load a pool laid out by cue_training's ``cue-train build-example-pool``."""

        directory = Path(directory)
        if (directory / POOL_DIRNAME / META_NAME).is_file():
            directory = directory / POOL_DIRNAME
        vectors = np.load(directory / VECTORS_NAME).astype(np.float32)
        index = EmbeddingIndex.load(directory / INDEX_NAME, int(vectors.shape[1]))
        session_ids: list[str] = []
        examples: list[list[PoolExample]] = []
        with (directory / META_NAME).open(encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                row = json.loads(line)
                session_ids.append(str(row["session_id"]))
                examples.append(
                    [PoolExample(text=str(e["text"]), kind=str(e["kind"])) for e in (row.get("examples") or [])]
                )
        fingerprint = ""
        manifest_path = directory / MANIFEST_NAME
        if manifest_path.is_file():
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            fingerprint = str(manifest.get("checkpoint_fingerprint") or "")
        log(TAG, f"loaded {len(session_ids)} pool sessions from {directory}")
        return cls(session_ids, vectors, examples, index, fingerprint)

    @classmethod
    def from_hub_dataset(cls, repo_id: str, *, split: str = "train") -> ExamplePool:
        """Load the pool from a dataset of {session_id, cue_embedding, examples}.

        Accepts a Hub dataset id or a ``save_to_disk`` directory.
        """

        try:
            from datasets import load_dataset, load_from_disk
        except ImportError as exc:  # pragma: no cover - dependency hint
            raise ImportError(
                "example-pool retrieval from the Hub needs `datasets`. "
                "Install with: pip install 'cue-simulator[retrieval]'"
            ) from exc
        if Path(repo_id).is_dir():
            dataset = load_from_disk(repo_id)
            if not hasattr(dataset, "column_names") or isinstance(dataset.column_names, dict):
                dataset = dataset[split]
        else:
            dataset = load_dataset(repo_id, split=split)
        session_ids = [str(x) for x in dataset["session_id"]]
        vectors = np.asarray(dataset["cue_embedding"], dtype=np.float32)
        examples = [
            [PoolExample(text=str(e["text"]), kind=str(e["kind"])) for e in (row or [])] for row in dataset["examples"]
        ]
        fingerprint = ""
        info = getattr(dataset, "info", None)
        if info is not None and getattr(info, "description", ""):
            fingerprint = _fingerprint_from_description(info.description)
        log(TAG, f"loaded {len(session_ids)} pool sessions from dataset {repo_id}")
        return cls.from_rows(session_ids, vectors, examples, checkpoint_fingerprint=fingerprint)


def _fingerprint_from_description(description: str) -> str:
    for line in description.splitlines():
        if line.lower().startswith("checkpoint_fingerprint:"):
            return line.split(":", 1)[1].strip()
    return ""


def inject_examples(
    manual: dict[str, Any] | None,
    *,
    pool: ExamplePool,
    cue_embedding: list[float] | np.ndarray,
    k_sessions: int = 8,
    n_general: int = 2,
    n_specific: int = 6,
    exclude_session_id: str | None = None,
) -> dict[str, Any] | None:
    """Replace a decoded manual's examples with pool neighbors of its CUE vector.

    Mirrors cue_training evaluation, which clears the general block's examples and fills the
    user-specific block from retrieval; after merging, that is the manual's example list.
    """

    if manual is None:
        return None
    retrieved = pool.retrieve(
        cue_embedding,
        k_sessions=k_sessions,
        n_general=n_general,
        n_specific=n_specific,
        exclude_session_id=exclude_session_id,
    )
    return {**manual, "examples": retrieved}
