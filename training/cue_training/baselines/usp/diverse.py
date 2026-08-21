"""Diverse Profile Sampling (USP paper Sec 4.3) over LMSYS-USP profiles.

Embeds profiles with SimCSE, reduces with UMAP (PCA fallback), fits a Gaussian KDE,
then samples naturalistic profiles. Virtual profiles mix objective-fact (OF) text from
one nearest neighbor with subjective-characteristic (SC) text from another.

LMSYS-USP profiles are natural-language strings with OF and SC separated by a blank
line (``\\n\\n``) — see https://huggingface.co/datasets/wangkevin02/LMSYS-USP.
"""

from __future__ import annotations

import os
from functools import lru_cache
from typing import Any, Sequence

import numpy as np

from cue_training.baselines.usp.extractor import profile_text
from cue_training.runlog.log import log, warn

DEFAULT_SIMCSE = "princeton-nlp/sup-simcse-roberta-large"
LMSYS_USP_DATASET = "wangkevin02/LMSYS-USP"
# Lean schema OF / SC keys (paired-arm extracts). LMSYS prose uses paragraph split.
OF_KEYS = ("goals", "knowledge_level", "preferences")
SC_KEYS = ("personality", "communication_style")
UMAP_DIM = int(os.environ.get("USP_DIVERSE_UMAP_DIM", "5"))
KDE_BANDWIDTH = float(os.environ.get("USP_DIVERSE_KDE_BANDWIDTH", "0.5"))
SYNTH_FRAC = float(os.environ.get("USP_DIVERSE_SYNTH_FRAC", "0.5"))
SIMCSE_BATCH = 128


def _env_pool_cap() -> int | None:
    """Optional hard cap via ``USP_LMSYS_POOL_CAP``; unset/empty/0 = full train split."""

    raw = os.environ.get("USP_LMSYS_POOL_CAP", "").strip()
    if not raw or raw.lower() in ("0", "none", "full", "all"):
        return None
    return max(1, int(raw))


LMSYS_POOL_CAP = _env_pool_cap()


class _SimCSE:
    """SimCSE sentence similarity via CLS pooling (transformers only)."""

    def __init__(self, model_name: str, device: str) -> None:
        import torch
        from transformers import AutoModel, AutoTokenizer

        self.torch = torch
        self.device = device if torch.cuda.is_available() else "cpu"
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name).to(self.device).eval()

    def embed(self, texts: list[str], batch_size: int = SIMCSE_BATCH):
        chunks = []
        for start in range(0, len(texts), max(1, batch_size)):
            batch = [t or "" for t in texts[start : start + max(1, batch_size)]]
            enc = self.tokenizer(
                batch, padding=True, truncation=True, max_length=256, return_tensors="pt"
            ).to(self.device)
            with self.torch.no_grad():
                out = self.model(**enc)
            chunks.append(self.torch.nn.functional.normalize(out.last_hidden_state[:, 0], dim=-1))
        if not chunks:
            hidden = int(getattr(self.model.config, "hidden_size", 1024))
            return self.torch.empty((0, hidden), device=self.device)
        return self.torch.cat(chunks, dim=0)


def _split_of_sc_text(profile: str) -> tuple[str, str]:
    parts = [p.strip() for p in str(profile).split("\n\n") if p.strip()]
    if not parts:
        return "", ""
    if len(parts) == 1:
        return parts[0], parts[0]
    return parts[0], "\n\n".join(parts[1:])


def _as_profile_dict(profile: dict[str, Any] | str) -> dict[str, Any]:
    if isinstance(profile, dict):
        # Prose LMSYS row wrapped as {"profile": "..."}.
        if "profile" in profile and not any(k in profile for k in OF_KEYS + SC_KEYS):
            text = str(profile.get("profile") or "")
            of, sc = _split_of_sc_text(text)
            return {
                "goals": of or "unspecified",
                "knowledge_level": "unspecified",
                "preferences": "unspecified",
                "personality": sc or of or "unspecified",
                "communication_style": "unspecified",
                "_full_text": text,
                "_of_text": of,
                "_sc_text": sc,
            }
        return dict(profile)
    of, sc = _split_of_sc_text(profile)
    return {
        "goals": of or "unspecified",
        "knowledge_level": "unspecified",
        "preferences": "unspecified",
        "personality": sc or of or "unspecified",
        "communication_style": "unspecified",
        "_full_text": str(profile),
        "_of_text": of,
        "_sc_text": sc,
    }


def _text(profile: dict[str, Any] | str) -> str:
    if isinstance(profile, str):
        return profile
    if profile.get("_full_text"):
        return str(profile["_full_text"])
    return profile_text(profile)


def _mix_of_sc(a: dict[str, Any], b: dict[str, Any]) -> dict[str, Any]:
    """Combine OF from ``a`` with SC from ``b`` (paper virtual-profile construction)."""

    a = _as_profile_dict(a)
    b = _as_profile_dict(b)
    # Prefer LMSYS paragraph mix when both carry OF/SC prose.
    if a.get("_of_text") is not None and b.get("_sc_text") is not None:
        mixed = f"{a.get('_of_text') or ''}\n\n{b.get('_sc_text') or ''}".strip()
        return _as_profile_dict(mixed)
    out = dict(a)
    for key in SC_KEYS:
        if key in b:
            out[key] = b[key]
    out.pop("_full_text", None)
    out.pop("_of_text", None)
    out.pop("_sc_text", None)
    return out


@lru_cache(maxsize=2)
def load_lmsys_usp_profiles(
    split: str = "train",
    *,
    limit: int | None = LMSYS_POOL_CAP,
) -> tuple[str, ...]:
    """Load LMSYS-USP profile strings from HuggingFace (cached).

    Default ``limit`` is ``None`` (full split). Set ``USP_LMSYS_POOL_CAP`` to a
    positive int only when you need a smaller pool for memory/runtime.
    """

    from datasets import load_dataset

    ds = load_dataset(LMSYS_USP_DATASET, split=split)
    profiles: list[str] = []
    for row in ds:
        text = str(row.get("profile") or "").strip()
        if text:
            profiles.append(text)
        if limit is not None and len(profiles) >= int(limit):
            break
    if not profiles:
        raise RuntimeError(f"{LMSYS_USP_DATASET} split={split!r} yielded no profiles")
    log("usp.diverse", f"loaded {len(profiles)} LMSYS-USP profiles (split={split})")
    return tuple(profiles)


class DiverseProfileSampler:
    """Fit on a profile pool, then sample (real or OF/SC-synthesized) profiles."""

    def __init__(
        self,
        profiles: Sequence[dict[str, Any] | str],
        *,
        seed: int = 0,
        simcse_model: str = DEFAULT_SIMCSE,
        device: str | None = None,
        synth_frac: float = SYNTH_FRAC,
    ) -> None:
        if not profiles:
            raise ValueError("DiverseProfileSampler needs a non-empty profile pool")
        self._rng = np.random.default_rng(seed)
        self._synth_frac = float(np.clip(synth_frac, 0.0, 1.0))
        self._profiles = [_as_profile_dict(p) for p in profiles]
        self._texts = [_text(p) for p in self._profiles]
        self._xy = self._fit_space(simcse_model=simcse_model, device=device)
        self._kde = self._fit_kde(self._xy)

    def _embed(self, texts: list[str], *, simcse_model: str, device: str | None) -> np.ndarray:
        enc = _SimCSE(simcse_model, device or ("cuda" if __import__("torch").cuda.is_available() else "cpu"))
        return enc.embed(texts).detach().cpu().numpy().astype(np.float64)

    def _fit_space(self, *, simcse_model: str, device: str | None) -> np.ndarray:
        emb = self._embed(self._texts, simcse_model=simcse_model, device=device)
        n, dim = emb.shape
        target = max(2, min(UMAP_DIM, n - 1, dim))
        try:
            import umap

            reducer = umap.UMAP(
                n_components=target,
                n_neighbors=min(15, max(2, n - 1)),
                min_dist=0.1,
                metric="cosine",
                random_state=0,
            )
            xy = reducer.fit_transform(emb)
            self._reducer = "umap"
            return np.asarray(xy, dtype=np.float64)
        except Exception as exc:  # noqa: BLE001
            warn("usp.diverse", f"UMAP unavailable ({exc}); falling back to PCA")
            from sklearn.decomposition import PCA

            xy = PCA(n_components=target, random_state=0).fit_transform(emb)
            self._reducer = "pca"
            return np.asarray(xy, dtype=np.float64)

    def _fit_kde(self, xy: np.ndarray):
        from sklearn.neighbors import KernelDensity

        kde = KernelDensity(bandwidth=KDE_BANDWIDTH, kernel="gaussian")
        kde.fit(xy)
        return kde

    def _nearest(self, point: np.ndarray, k: int = 2) -> list[int]:
        dists = np.linalg.norm(self._xy - point[None, :], axis=1)
        order = np.argsort(dists)
        return [int(i) for i in order[: max(1, min(k, len(order)))]]

    def sample(self) -> dict[str, Any]:
        seed = int(self._rng.integers(0, 2**31 - 1))
        point = self._kde.sample(1, random_state=seed)[0]
        idxs = self._nearest(point, k=2)
        if len(idxs) == 1 or self._rng.random() >= self._synth_frac:
            return dict(self._profiles[idxs[0]])
        return _mix_of_sc(self._profiles[idxs[0]], self._profiles[idxs[1]])

    def sample_text(self) -> str:
        return _text(self.sample())

    def sample_many(self, n: int) -> list[dict[str, Any]]:
        return [self.sample() for _ in range(max(0, int(n)))]


def build_diverse_texts(
    profiles: Sequence[dict[str, Any] | str],
    n: int,
    *,
    seed: int = 0,
) -> list[str]:
    """Convenience: ``n`` diverse profile texts from ``profiles`` (logs reducer used)."""

    if not profiles:
        return []
    if len(profiles) == 1:
        return [_text(profiles[0])] * n
    sampler = DiverseProfileSampler(profiles, seed=seed)
    log("usp.diverse", f"sampler ready ({len(profiles)} pool, reducer={sampler._reducer})")
    return [sampler.sample_text() for _ in range(n)]


def build_diverse_texts_from_lmsys(
    n: int,
    *,
    seed: int = 0,
    split: str = "train",
    limit: int | None = LMSYS_POOL_CAP,
) -> list[str]:
    """Diverse Profile Sampling over the published LMSYS-USP training profiles."""

    profiles = load_lmsys_usp_profiles(split, limit=limit)
    return build_diverse_texts(profiles, n, seed=seed)
