"""USP authenticity metrics (SimCSE / Wegmann / AVA) without downloading models."""

from __future__ import annotations

import numpy as np

from cue_training.evaluation.metrics.data import Episode
from cue_training.evaluation.metrics.metrics import authenticity as A
from cue_training.evaluation.metrics.run import ALL_METRICS, DEFAULT_METRICS, _parse_metrics


def _ep(eid: str, human: list[str], proxy: list[str]) -> Episode:
    return Episode(
        episode_id=eid,
        human=[{"role": "user", "content": t} for t in human]
        + [{"role": "assistant", "content": "ok"}],
        base=[{"role": "user", "content": "x"}],
        proxy=[{"role": "user", "content": t} for t in proxy]
        + [{"role": "assistant", "content": "ok"}],
        metadata={},
        base_metadata={},
        arm="paired",
        domain="airline",
    )


class _FakeEncoder:
    """Deterministic unit vectors from text length parity (even/odd)."""

    def encode(self, texts: list[str]) -> np.ndarray:
        rows = []
        for t in texts:
            # Same first char → identical vector; else orthogonal-ish.
            v = np.zeros(4, dtype=np.float32)
            if not t:
                rows.append(v)
                continue
            idx = (ord(t[0]) + len(t)) % 4
            v[idx] = 1.0
            rows.append(v)
        return np.stack(rows, axis=0) if rows else np.zeros((0, 4), dtype=np.float32)


def test_default_keeps_wegmann_ava():
    assert "mimicry/wegmann_ava" in DEFAULT_METRICS
    assert "authenticity/ava" not in DEFAULT_METRICS
    assert "authenticity/ava" not in ALL_METRICS
    assert set(DEFAULT_METRICS) == set(ALL_METRICS)


def test_ava_alias_expands():
    assert _parse_metrics("ava") == ["mimicry/wegmann_ava"]


def test_utterance_and_dialogue_sims(monkeypatch):
    fake = _FakeEncoder()
    monkeypatch.setattr(A, "get_simcse_encoder", lambda *a, **k: fake)
    monkeypatch.setattr(A, "get_style_encoder", lambda *a, **k: fake)
    episodes = [
        _ep("a", ["hello", "world"], ["hello", "world"]),
        _ep("b", ["zzzz"], ["yyyy"]),
    ]
    utt = A.authenticity_sem_sim_utt(episodes)
    assert utt.sample_size == 3  # 2 + 1 paired turns
    assert utt.mean is not None
    # First episode both turns match → cosine 1; second differs → 0.
    assert abs(utt.mean - (200.0 / 3.0)) < 1e-3

    dia = A.authenticity_style_sim_dia(episodes)
    assert dia.sample_size == 2
    # Concat "hello\nworld" vs same → 1; "zzzz" vs "yyyy" → 0 → mean 50.
    assert abs(dia.mean - 50.0) < 1e-6


def test_ava_threshold_and_hits(monkeypatch):
    fake = _FakeEncoder()
    monkeypatch.setattr(A, "get_style_encoder", lambda *a, **k: fake)
    episodes = [
        _ep("a", ["hello", "hello2"], ["hello", "hello2"]),
        _ep("b", ["zzzz"], ["zzzz"]),
    ]
    ava = A.authenticity_ava(episodes, threshold=0.5)
    assert ava.sample_size == 2
    assert ava.mean == 100.0
    assert ava.extras["threshold"] == 0.5


def test_backend_ava_and_raw_cosine_share_scores(monkeypatch):
    fake = _FakeEncoder()
    monkeypatch.setattr(
        A,
        "_backend_encoder",
        lambda backend, **_kwargs: (fake, f"{backend}-model", backend),
    )
    episodes = [
        _ep("a", ["hello"], ["hello"]),
        _ep("b", ["zzzz"], ["yyyy"]),
    ]

    cosine = A.mimicry_similarity(episodes, backend="simcse", kind="cosine")
    ava = A.mimicry_similarity(
        episodes,
        backend="simcse",
        kind="ava",
        threshold=0.5,
    )

    assert cosine.metric_name == "mimicry/simcse_cosine"
    assert cosine.mean == 0.5
    assert cosine.extras["unit"] == "cosine"
    assert ava.metric_name == "mimicry/simcse_ava"
    assert ava.mean == 50.0
    assert ava.extras["mean_cosine"] == 0.5
    assert ava.extras["threshold"] == 0.5
