"""Tests for diffusion-sampled CUE rollout arms."""

from __future__ import annotations

import numpy as np
import pytest

from cue_training.evaluation.common.sampler_arms import (
    PRECOMPUTED_CUE_ARMS,
    SAMPLED_ARMS,
    generate_sampled_arm_embeddings,
    has_sampled_arms,
    needs_precomputed_embeddings,
)


def test_sampled_arm_constants():
    assert "sampled_uncond" in SAMPLED_ARMS
    assert "sampled_pop" in SAMPLED_ARMS
    assert PRECOMPUTED_CUE_ARMS >= set(SAMPLED_ARMS)


def test_needs_precomputed_embeddings():
    assert not needs_precomputed_embeddings(["paired", "sample_shuffled"])
    assert needs_precomputed_embeddings(["paired", "sampled_uncond"])
    assert needs_precomputed_embeddings(["paired"], recipe_needs=True)


def test_generate_sampled_arm_embeddings_one_per_user(monkeypatch):
    calls: list[dict] = []

    def fake_sample(checkpoint, *, n, condition_embeddings=None, seed=0, **kwargs):
        calls.append(
            {
                "n": n,
                "cond": None if condition_embeddings is None else np.asarray(condition_embeddings).shape,
                "seed": seed,
                "guidance_w": kwargs.get("guidance_w"),
            }
        )
        return {"embeddings": np.arange(n * 4, dtype=np.float32).reshape(n, 4) + seed}

    eids = ["a1", "a2", "b1"]
    domains = {"a1": "airline", "a2": "airline", "b1": "retail"}
    by_key = {
        "a1": [1.0, 0.0, 0.0, 0.0],
        "a2": [0.0, 1.0, 0.0, 0.0],
        "b1": [0.0, 0.0, 1.0, 0.0],
    }
    out = generate_sampled_arm_embeddings(
        episode_ids=eids,
        domains=domains,
        by_key=by_key,
        arms=("sampled_uncond", "sampled_pop", "paired"),
        sampler_checkpoint="/tmp/sampler",
        seed=0,
        sample_fn=fake_sample,
    )
    assert set(out) == {"sampled_uncond", "sampled_pop"}
    assert set(out["sampled_uncond"]) == set(eids)
    assert set(out["sampled_pop"]) == set(eids)
    # One call per (domain, arm): airline x2 + retail x2
    assert len(calls) == 4
    uncond = [c for c in calls if c["cond"] is None]
    pop = [c for c in calls if c["cond"] is not None]
    assert len(uncond) == 2 and len(pop) == 2
    assert {c["n"] for c in uncond} == {2, 1}  # airline=2, retail=1
    # Per-sample random sets: [n, k_eff, D] with k_eff = min(8, pop_n)
    assert {c["cond"] for c in pop} == {(2, 2, 4), (1, 1, 4)}


def test_sampled_pop_per_sample_random_sets():
    seen: list[np.ndarray] = []

    def fake_sample(checkpoint, *, n, condition_embeddings=None, seed=0, **kwargs):
        assert condition_embeddings is not None
        cond = np.asarray(condition_embeddings, dtype=np.float32)
        seen.append(cond)
        return {"embeddings": np.zeros((n, cond.shape[-1]), dtype=np.float32)}

    # 10 population members → each of 5 episodes gets a distinct random 8-set
    eids = [f"e{i}" for i in range(5)]
    domains = {e: "airline" for e in eids}
    by_key = {e: np.eye(10, dtype=np.float32)[i].tolist() for i, e in enumerate(eids)}
    # pad population via extra keys not in episode_ids? use all eids as pop of 5, k=3
    out = generate_sampled_arm_embeddings(
        episode_ids=eids,
        domains=domains,
        by_key=by_key,
        arms=("sampled_pop",),
        sampler_checkpoint="/tmp/sampler",
        seed=0,
        pop_max_k=3,
        sample_fn=fake_sample,
    )
    assert len(out["sampled_pop"]) == 5
    assert len(seen) == 1
    cond = seen[0]
    assert cond.shape == (5, 3, 10)
    # rows should not all be identical (randomization per sample)
    assert not all(np.allclose(cond[0], cond[i]) for i in range(1, 5))


def test_generate_requires_checkpoint():
    with pytest.raises(ValueError, match="sampler_checkpoint"):
        generate_sampled_arm_embeddings(
            episode_ids=["a"],
            domains={"a": "x"},
            by_key={"a": [0.0, 1.0]},
            arms=("sampled_uncond",),
            sampler_checkpoint="",
            sample_fn=lambda *a, **k: {"embeddings": np.zeros((1, 2))},
        )


def test_has_sampled_arms():
    assert has_sampled_arms(["paired", "sampled_pop"])
    assert not has_sampled_arms(["paired", "sample_shuffled"])


def test_coverage_includes_sampled_arms():
    from cue_training.evaluation.metrics.policy import SAMPLED_ARMS as cov

    assert "sampled_uncond" in cov
    assert "sampled_pop" in cov
