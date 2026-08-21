"""FAISS IndexFlatIP wrapper with a NumPy fallback for non-Linux / tests."""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Any

import numpy as np

from cue_hf.log import warn

TAG = "index"


def _try_import_faiss() -> Any | None:
    # Official dep is Linux-only; macOS/Py3.13 wheels often segfault in IndexFlatIP.search.
    if sys.platform != "linux" and os.environ.get("CUE_FORCE_FAISS", "") != "1":
        return None
    try:
        import faiss  # type: ignore

        return faiss
    except ImportError:
        return None


_WARNED_FALLBACK = False


def require_faiss_or_fallback() -> tuple[Any | None, str]:
    """Return (faiss_module_or_None, backend_name)."""

    global _WARNED_FALLBACK
    faiss = _try_import_faiss()
    if faiss is not None:
        return faiss, "faiss"
    if not _WARNED_FALLBACK:
        warn(
            TAG,
            "faiss-cpu not available (or non-Linux); using NumPy brute-force IP index "
            "(install faiss-cpu on Linux for large pools; set CUE_FORCE_FAISS=1 to override)",
        )
        _WARNED_FALLBACK = True
    return None, "numpy"


class EmbeddingIndex:
    """L2-normalized vectors, cosine via inner product."""

    def __init__(self, dim: int, *, backend: str | None = None) -> None:
        self.dim = int(dim)
        self._faiss, detected = require_faiss_or_fallback()
        self.backend = backend or detected
        self._vectors: np.ndarray | None = None
        self._index: Any | None = None
        if self.backend == "faiss" and self._faiss is not None:
            self._index = self._faiss.IndexFlatIP(self.dim)
        else:
            self.backend = "numpy"
            self._vectors = np.zeros((0, self.dim), dtype=np.float32)

    @property
    def ntotal(self) -> int:
        if self.backend == "faiss" and self._index is not None:
            return int(self._index.ntotal)
        assert self._vectors is not None
        return int(self._vectors.shape[0])

    def add(self, vectors: np.ndarray) -> None:
        vecs = np.ascontiguousarray(_l2_normalize(np.asarray(vectors, dtype=np.float32)))
        if vecs.ndim != 2 or vecs.shape[1] != self.dim:
            raise ValueError(f"expected [N,{self.dim}], got {vecs.shape}")
        if self.backend == "faiss" and self._index is not None:
            self._index.add(vecs)
        else:
            assert self._vectors is not None
            self._vectors = vecs if self._vectors.shape[0] == 0 else np.vstack([self._vectors, vecs])

    def search(self, queries: np.ndarray, k: int) -> tuple[np.ndarray, np.ndarray]:
        q = np.ascontiguousarray(_l2_normalize(np.asarray(queries, dtype=np.float32)))
        k = max(1, min(int(k), max(1, self.ntotal)))
        if self.ntotal == 0:
            empty_d = np.full((q.shape[0], k), -np.inf, dtype=np.float32)
            empty_i = np.full((q.shape[0], k), -1, dtype=np.int64)
            return empty_d, empty_i
        if self.backend == "faiss" and self._index is not None:
            return self._index.search(q, k)
        assert self._vectors is not None
        sims = q @ self._vectors.T
        # partial top-k
        idx = np.argpartition(-sims, kth=min(k, sims.shape[1]) - 1, axis=1)[:, :k]
        row = np.arange(q.shape[0])[:, None]
        part_sims = sims[row, idx]
        order = np.argsort(-part_sims, axis=1)
        top_idx = idx[row, order]
        top_sims = part_sims[row, order]
        return top_sims.astype(np.float32), top_idx.astype(np.int64)

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        backend_path = Path(str(path) + ".backend")
        if self.backend == "faiss" and self._index is not None and self._faiss is not None:
            self._faiss.write_index(self._index, str(path))
            backend_path.write_text("faiss\n", encoding="utf-8")
            return
        assert self._vectors is not None
        np.save(Path(str(path) + ".npy"), self._vectors)
        backend_path.write_text("numpy\n", encoding="utf-8")

    @classmethod
    def load(cls, path: str | Path, dim: int) -> EmbeddingIndex:
        path = Path(path)
        backend_path = Path(str(path) + ".backend")
        backend = "faiss"
        if backend_path.is_file():
            backend = backend_path.read_text(encoding="utf-8").strip() or "faiss"
        npy_path = Path(str(path) + ".npy")
        if backend != "faiss" or (not path.is_file() and npy_path.is_file()):
            backend = "numpy"
        obj = cls(dim, backend=backend)
        if backend == "faiss":
            faiss = _try_import_faiss()
            if faiss is None:
                raise ImportError(f"Index at {path} requires faiss-cpu (Linux). Install with: pip install faiss-cpu")
            obj._faiss = faiss
            obj._index = faiss.read_index(str(path))
            obj.backend = "faiss"
            return obj
        if not npy_path.is_file():
            raise FileNotFoundError(f"numpy index vectors missing: {npy_path}")
        obj._vectors = np.load(npy_path).astype(np.float32)
        obj.backend = "numpy"
        return obj


def _l2_normalize(x: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    norms = np.linalg.norm(x, axis=-1, keepdims=True)
    return (x / np.maximum(norms, eps)).astype(np.float32)
