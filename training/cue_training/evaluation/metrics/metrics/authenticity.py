"""USP-style authenticity metrics (Wang et al., ACL 2025 / arXiv:2502.18968).

Authenticity follows the paper:
  - Sem-Sim via SimCSE (Gao et al., 2021)
  - Style-Sim via style embeddings (Wegmann et al., 2022)
  - AVA (Author Verification Accuracy) on style cosine vs a threshold

``D_utt`` pairs human/proxy user utterances in order (min length).
``D_dia`` concatenates all user utterances per dialogue, then compares.
Scores are reported as percentages (cosine × 100) to match the USP tables.
"""

from __future__ import annotations

from functools import lru_cache
from typing import Any

import numpy as np

from cue_training.evaluation.metrics.data import Episode, user_text, user_turn_texts
from cue_training.evaluation.metrics.stats import Aggregate, aggregate
from cue_training.runlog.log import warn

DEFAULT_SIMCSE_MODEL = "princeton-nlp/sup-simcse-roberta-large"
DEFAULT_STYLE_MODEL = "AnnaWegmann/Style-Embedding"
# Fallback when human-corpus calibration has too few pairs.
DEFAULT_AVA_THRESHOLD = 0.5


class SimCSEEncoder:
    """CLS-pooled SimCSE encoder (L2-normalized), matching USP / RLCC usage."""

    def __init__(
        self,
        model_name: str = DEFAULT_SIMCSE_MODEL,
        *,
        device: str | None = None,
        batch_size: int = 32,
        max_length: int = 256,
    ) -> None:
        import torch
        from transformers import AutoModel, AutoTokenizer

        self.torch = torch
        self.batch_size = batch_size
        self.max_length = max_length
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model_name = model_name
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name).to(self.device).eval()

    def encode(self, texts: list[str]) -> np.ndarray:
        torch = self.torch
        vectors: list[np.ndarray] = []
        for start in range(0, len(texts), self.batch_size):
            chunk = [t or "" for t in texts[start : start + self.batch_size]]
            enc = self.tokenizer(
                chunk,
                padding=True,
                truncation=True,
                max_length=self.max_length,
                return_tensors="pt",
            ).to(self.device)
            with torch.no_grad():
                out = self.model(**enc)
            cls = torch.nn.functional.normalize(out.last_hidden_state[:, 0].float(), dim=-1)
            vectors.append(cls.cpu().numpy())
        if not vectors:
            return np.zeros((0, 1), dtype=np.float32)
        return np.concatenate(vectors, axis=0)


@lru_cache(maxsize=2)
def get_simcse_encoder(
    model_name: str = DEFAULT_SIMCSE_MODEL,
    *,
    batch_size: int = 32,
) -> SimCSEEncoder:
    return SimCSEEncoder(model_name, batch_size=batch_size)


@lru_cache(maxsize=2)
def get_style_encoder(
    model_name: str = DEFAULT_STYLE_MODEL,
    *,
    batch_size: int = 16,
) -> Any:
    from cue_training.evaluation.metrics.metrics.style_encoder import get_encoder

    return get_encoder(model_name, batch_size=batch_size)


def _pair_cosines(a: np.ndarray, b: np.ndarray) -> list[float]:
    if len(a) == 0 or len(b) == 0 or len(a) != len(b):
        return []
    dots = (a * b).sum(axis=1)
    denom = np.linalg.norm(a, axis=1) * np.linalg.norm(b, axis=1)
    return [float(v) for v in np.divide(dots, denom, out=np.zeros_like(dots), where=denom > 0)]


def _pct(vals: list[float]) -> list[float]:
    return [100.0 * float(v) for v in vals]


def _utterance_pairs(episodes: list[Episode]) -> tuple[list[str], list[str]]:
    left: list[str] = []
    right: list[str] = []
    for ep in episodes:
        h = user_turn_texts(ep.human)
        p = user_turn_texts(ep.proxy)
        n = min(len(h), len(p))
        left.extend(h[:n])
        right.extend(p[:n])
    return left, right


def _dialogue_pairs(episodes: list[Episode]) -> tuple[list[str], list[str]]:
    left = [user_text(ep.human) for ep in episodes]
    right = [user_text(ep.proxy) for ep in episodes]
    keep = [(a, b) for a, b in zip(left, right) if a.strip() and b.strip()]
    if not keep:
        return [], []
    a, b = zip(*keep)
    return list(a), list(b)


def _calibrate_ava_threshold(style_encoder: Any, episodes: list[Episode]) -> float:
    """Pick a same-author threshold from human style pairs (USP/Wegmann AVA).

    Positives: consecutive user turns within the same human dialogue.
    Negatives: first user turns from different episodes.
    Threshold = midpoint of mean(pos) and mean(neg), clamped to a sane range.
    """

    pos_left: list[str] = []
    pos_right: list[str] = []
    first_turns: list[str] = []
    for ep in episodes:
        turns = user_turn_texts(ep.human)
        if turns:
            first_turns.append(turns[0])
        for i in range(len(turns) - 1):
            pos_left.append(turns[i])
            pos_right.append(turns[i + 1])
    pos_sims: list[float] = []
    if pos_left:
        ea = style_encoder.encode(pos_left)
        eb = style_encoder.encode(pos_right)
        pos_sims = _pair_cosines(ea, eb)
    neg_sims: list[float] = []
    if len(first_turns) >= 2:
        left = first_turns
        right = first_turns[1:] + first_turns[:1]
        ea = style_encoder.encode(left)
        eb = style_encoder.encode(right)
        # Drop accidental self-pairs when only one unique episode.
        neg_sims = [
            s for i, s in enumerate(_pair_cosines(ea, eb)) if left[i] != right[i]
        ]
    if pos_sims and neg_sims:
        thr = 0.5 * (float(np.mean(pos_sims)) + float(np.mean(neg_sims)))
        return float(np.clip(thr, 0.05, 0.95))
    if pos_sims:
        return float(np.clip(np.mean(pos_sims) - 0.05, 0.05, 0.95))
    return DEFAULT_AVA_THRESHOLD


def _encode_pair_sims(
    encoder: Any,
    left: list[str],
    right: list[str],
) -> list[float]:
    if not left:
        return []
    return _pair_cosines(encoder.encode(left), encoder.encode(right))


def authenticity_sem_sim_utt(
    episodes: list[Episode],
    *,
    model_name: str = DEFAULT_SIMCSE_MODEL,
) -> Aggregate:
    """Utterance-level semantic similarity (SimCSE) × 100."""

    left, right = _utterance_pairs(episodes)
    if not left:
        return aggregate("authenticity/sem_sim_utt", [], {"skipped": "no paired user utterances"})
    try:
        sims = _encode_pair_sims(get_simcse_encoder(model_name), left, right)
    except Exception as exc:  # noqa: BLE001
        warn("metrics", f"SimCSE utterance encode failed: {exc}")
        return aggregate("authenticity/sem_sim_utt", [], {"error": str(exc)})
    return aggregate(
        "authenticity/sem_sim_utt",
        _pct(sims),
        {
            "model_name": model_name,
            "level": "utterance",
            "space": "simcse",
            "unit": "percent",
            "paper": "wang2025usp",
            "n_pairs": len(sims),
        },
    )


def authenticity_style_sim_utt(
    episodes: list[Episode],
    *,
    model_name: str = DEFAULT_STYLE_MODEL,
) -> Aggregate:
    """Utterance-level stylistic similarity (Wegmann) × 100."""

    left, right = _utterance_pairs(episodes)
    if not left:
        return aggregate("authenticity/style_sim_utt", [], {"skipped": "no paired user utterances"})
    try:
        sims = _encode_pair_sims(get_style_encoder(model_name), left, right)
    except Exception as exc:  # noqa: BLE001
        warn("metrics", f"style embedding utterance encode failed: {exc}")
        return aggregate("authenticity/style_sim_utt", [], {"error": str(exc)})
    return aggregate(
        "authenticity/style_sim_utt",
        _pct(sims),
        {
            "model_name": model_name,
            "level": "utterance",
            "space": "wegmann_style",
            "unit": "percent",
            "paper": "wang2025usp",
            "n_pairs": len(sims),
        },
    )


def authenticity_sem_sim_dia(
    episodes: list[Episode],
    *,
    model_name: str = DEFAULT_SIMCSE_MODEL,
) -> Aggregate:
    """Dialogue-level semantic similarity (concatenated user utterances) × 100."""

    left, right = _dialogue_pairs(episodes)
    if not left:
        return aggregate("authenticity/sem_sim_dia", [], {"skipped": "no paired dialogues"})
    try:
        sims = _encode_pair_sims(get_simcse_encoder(model_name), left, right)
    except Exception as exc:  # noqa: BLE001
        warn("metrics", f"SimCSE dialogue encode failed: {exc}")
        return aggregate("authenticity/sem_sim_dia", [], {"error": str(exc)})
    return aggregate(
        "authenticity/sem_sim_dia",
        _pct(sims),
        {
            "model_name": model_name,
            "level": "dialogue",
            "space": "simcse",
            "unit": "percent",
            "paper": "wang2025usp",
            "aggregation": "concat_user_utterances",
            "n_pairs": len(sims),
        },
    )


def authenticity_style_sim_dia(
    episodes: list[Episode],
    *,
    model_name: str = DEFAULT_STYLE_MODEL,
) -> Aggregate:
    """Dialogue-level stylistic similarity (concatenated user utterances) × 100."""

    left, right = _dialogue_pairs(episodes)
    if not left:
        return aggregate("authenticity/style_sim_dia", [], {"skipped": "no paired dialogues"})
    try:
        sims = _encode_pair_sims(get_style_encoder(model_name), left, right)
    except Exception as exc:  # noqa: BLE001
        warn("metrics", f"style embedding dialogue encode failed: {exc}")
        return aggregate("authenticity/style_sim_dia", [], {"error": str(exc)})
    return aggregate(
        "authenticity/style_sim_dia",
        _pct(sims),
        {
            "model_name": model_name,
            "level": "dialogue",
            "space": "wegmann_style",
            "unit": "percent",
            "paper": "wang2025usp",
            "aggregation": "concat_user_utterances",
            "n_pairs": len(sims),
        },
    )


def authenticity_ava(
    episodes: list[Episode],
    *,
    model_name: str = DEFAULT_STYLE_MODEL,
    threshold: float | None = None,
) -> Aggregate:
    """Author Verification Accuracy on dialogue-level style pairs (percent).

    A human/proxy dialogue pair is counted correct when style cosine ≥ threshold
    (predicted same author). Threshold is calibrated on human consecutive-turn
    positives vs cross-episode negatives when ``threshold`` is None.
    """

    left, right = _dialogue_pairs(episodes)
    if not left:
        return aggregate("authenticity/ava", [], {"skipped": "no paired dialogues"})
    try:
        encoder = get_style_encoder(model_name)
        sims = _encode_pair_sims(encoder, left, right)
        thr = float(threshold) if threshold is not None else _calibrate_ava_threshold(encoder, episodes)
    except Exception as exc:  # noqa: BLE001
        warn("metrics", f"AVA encode failed: {exc}")
        return aggregate("authenticity/ava", [], {"error": str(exc)})
    hits = [100.0 if s >= thr else 0.0 for s in sims]
    return aggregate(
        "authenticity/ava",
        hits,
        {
            "model_name": model_name,
            "level": "dialogue",
            "space": "wegmann_style",
            "unit": "percent",
            "paper": "wang2025usp",
            "threshold": round(thr, 4),
            "threshold_source": "calibrated_human_pairs" if threshold is None else "explicit",
            "mean_style_cosine": round(float(np.mean(sims)), 4) if sims else None,
            "n_pairs": len(sims),
        },
    )


MIMICRY_BACKENDS = ("sim2real", "styledistance", "wegmann", "luar", "simcse")
MIMICRY_SIMILARITY_METRICS = tuple(
    f"mimicry/{backend}_{suffix}"
    for backend in MIMICRY_BACKENDS
    for suffix in ("ava", "cosine")
)


def _backend_encoder(
    backend: str, *, luar_model: str | None = None
) -> tuple[Any, str, str]:
    """Return an ``encode(texts)`` adapter, model label, and space label."""

    if backend == "sim2real":
        from cue_training.evaluation.metrics.metrics.local import _feature_vector

        class _Sim2RealEncoder:
            @staticmethod
            def encode(texts: list[str]) -> np.ndarray:
                return np.asarray([_feature_vector(text) for text in texts], dtype=np.float32)

        return _Sim2RealEncoder(), "sim2real_lexical_vector", "sim2real_vector"
    if backend == "styledistance":
        from cue_training.evaluation.metrics.metrics.style_encoder import (
            DEFAULT_STYLEDISTANCE_MODEL,
            get_styledistance_encoder,
        )

        return (
            get_styledistance_encoder(DEFAULT_STYLEDISTANCE_MODEL),
            DEFAULT_STYLEDISTANCE_MODEL,
            "styledistance",
        )
    if backend == "wegmann":
        return get_style_encoder(DEFAULT_STYLE_MODEL), DEFAULT_STYLE_MODEL, "wegmann_style"
    if backend == "luar":
        from cue_training.evaluation.metrics.metrics.luar_encoder import DEFAULT_LUAR_MODEL, get_luar_encoder

        model_name = luar_model or DEFAULT_LUAR_MODEL
        encoder = get_luar_encoder(model_name)

        class _LUAREncoder:
            @staticmethod
            def encode(texts: list[str]) -> np.ndarray:
                return np.asarray(encoder.encode_authors([[text] for text in texts]), dtype=np.float32)

        return _LUAREncoder(), model_name, "luar_author"
    if backend == "simcse":
        return get_simcse_encoder(DEFAULT_SIMCSE_MODEL), DEFAULT_SIMCSE_MODEL, "simcse"
    raise ValueError(f"unsupported mimicry backend: {backend}")


def mimicry_similarity(
    episodes: list[Episode],
    *,
    backend: str,
    kind: str,
    threshold: float | None = None,
    luar_model: str | None = None,
) -> Aggregate:
    """Dialogue-level AVA or raw cosine with a swappable embedding backend."""

    metric_name = f"mimicry/{backend}_{kind}"
    left, right = _dialogue_pairs(episodes)
    if not left:
        return aggregate(metric_name, [], {"skipped": "no paired dialogues"})
    try:
        encoder, model_name, space = _backend_encoder(backend, luar_model=luar_model)
        sims = _encode_pair_sims(encoder, left, right)
        thr = (
            float(threshold)
            if threshold is not None
            else _calibrate_ava_threshold(encoder, episodes)
        )
    except Exception as exc:  # noqa: BLE001
        warn("metrics", f"{metric_name} failed: {exc}")
        return aggregate(metric_name, [], {"error": str(exc)})
    common = {
        "model_name": model_name,
        "level": "dialogue",
        "space": space,
        "aggregation": "concat_user_utterances",
        "n_pairs": len(sims),
        "mean_cosine": round(float(np.mean(sims)), 4) if sims else None,
    }
    if kind == "cosine":
        return aggregate(metric_name, sims, {**common, "unit": "cosine"})
    if kind != "ava":
        raise ValueError(f"unsupported mimicry similarity kind: {kind}")
    hits = [100.0 if score >= thr else 0.0 for score in sims]
    return aggregate(
        metric_name,
        hits,
        {
            **common,
            "unit": "percent",
            "threshold": round(thr, 4),
            "threshold_source": "calibrated_human_pairs" if threshold is None else "explicit",
        },
    )


AUTHENTICITY_METRICS = (
    "authenticity/sem_sim_utt",
    "authenticity/style_sim_utt",
    "authenticity/sem_sim_dia",
    "authenticity/style_sim_dia",
    "authenticity/ava",
    *MIMICRY_SIMILARITY_METRICS,
)


def run_authenticity_metric(
    name: str,
    episodes: list[Episode],
    *,
    luar_model: str | None = None,
) -> Aggregate:
    if name in MIMICRY_SIMILARITY_METRICS:
        backend, kind = name.removeprefix("mimicry/").rsplit("_", 1)
        return mimicry_similarity(
            episodes,
            backend=backend,
            kind=kind,
            luar_model=luar_model,
        )
    if name == "authenticity/sem_sim_utt":
        return authenticity_sem_sim_utt(episodes)
    if name == "authenticity/style_sim_utt":
        return authenticity_style_sim_utt(episodes)
    if name == "authenticity/sem_sim_dia":
        return authenticity_sem_sim_dia(episodes)
    if name == "authenticity/style_sim_dia":
        return authenticity_style_sim_dia(episodes)
    if name == "authenticity/ava":
        # Alias → fixed Wegmann cosine AVA (never the legacy unnormalized-dot path).
        agg = mimicry_similarity(
            episodes,
            backend="wegmann",
            kind="ava",
            luar_model=luar_model,
        )
        return Aggregate(
            metric_name="authenticity/ava",
            mean=agg.mean,
            standard_deviation=agg.standard_deviation,
            confidence_interval=agg.confidence_interval,
            sample_size=agg.sample_size,
            extras={**(agg.extras or {}), "alias_of": "mimicry/wegmann_ava"},
        )
    return Aggregate(metric_name=name, mean=float("nan"), sample_size=0, extras={"error": "unknown authenticity metric"})
