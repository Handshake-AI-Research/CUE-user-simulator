"""USP Diverse Profile Sampling arm (no model downloads)."""

from __future__ import annotations

import numpy as np

from pathlib import Path

import pytest

from cue_training.baselines.usp import diverse as D
from cue_training.evaluation.metrics.policy import SAMPLED_ARMS
from cue_training.evaluation.rollouts.recipes import get_recipe


def test_resolve_profiles_dir_defaults_to_shipped(tmp_path: Path):
    from cue_training.evaluation.simulatorarena.data import default_profiles_dir, resolve_profiles_dir

    shipped = default_profiles_dir("writing")
    if shipped is None:
        pytest.skip("SimulatorArena profile files are not vendored")
    assert shipped.is_dir()
    assert (shipped / "writing_style.json").is_file()
    assert resolve_profiles_dir(None, "writing") == shipped
    assert resolve_profiles_dir(None, "math") is not None
    missing = tmp_path / "nope"
    assert resolve_profiles_dir(missing, "writing") is None
    assert resolve_profiles_dir(shipped, "writing") == shipped


def test_usp_recipe_includes_sample_diverse():
    recipe = get_recipe("baseline", "usp")
    assert recipe.default_arms == ("paired", "sample_diverse")


def test_coverage_sampled_arms_include_diverse():
    assert "sample_diverse" in SAMPLED_ARMS


def test_diverse_of_sc_mix():
    a = {
        "goals": "refund",
        "knowledge_level": "novice",
        "preferences": "email",
        "personality": "impatient",
        "communication_style": "terse",
    }
    b = {
        "goals": "upgrade",
        "knowledge_level": "expert",
        "preferences": "chat",
        "personality": "polite",
        "communication_style": "verbose",
    }
    mixed = D._mix_of_sc(a, b)
    assert mixed["goals"] == "refund"
    assert mixed["knowledge_level"] == "novice"
    assert mixed["preferences"] == "email"
    assert mixed["personality"] == "polite"
    assert mixed["communication_style"] == "verbose"


def test_sidecar_covers_arms(tmp_path):
    from cue_training.evaluation.common.io import write_jsonl
    from cue_training.evaluation.tau2_bench.personas import sidecar_covers_arms

    path = tmp_path / "personas.jsonl"
    assert not sidecar_covers_arms(path, "baseline:usp", ["paired", "sample_diverse"])
    write_jsonl(
        str(path),
        [
            {"variant": "baseline:usp", "arm": "paired", "episode_id": "e1", "persona": "a"},
            {"variant": "baseline:usp", "arm": "sample_shuffled", "episode_id": "e1", "persona": "b"},
        ],
    )
    assert sidecar_covers_arms(path, "baseline:usp", ["paired", "sample_shuffled"])
    assert not sidecar_covers_arms(path, "baseline:usp", ["paired", "sample_diverse"])
    write_jsonl(
        str(path),
        [
            {"variant": "baseline:usp", "arm": "paired", "episode_id": "e1", "persona": "a"},
            {"variant": "baseline:usp", "arm": "sample_shuffled", "episode_id": "e1", "persona": "b"},
            {"variant": "baseline:usp", "arm": "sample_diverse", "episode_id": "e1", "persona": "c"},
        ],
    )
    assert sidecar_covers_arms(path, "baseline:usp", ["paired", "sample_shuffled", "sample_diverse"])


def test_diverse_sampler_with_fake_embeds(monkeypatch):
    profiles = [
        {
            "goals": f"g{i}",
            "knowledge_level": "novice",
            "preferences": "chat",
            "personality": f"p{i}",
            "communication_style": "terse",
        }
        for i in range(6)
    ]

    def _fake_embed(self, texts, *, simcse_model, device):  # noqa: ANN001
        # Distinct points so KDE/nearest-neighbor are well-defined.
        return np.eye(len(texts), 8, dtype=np.float64) + 0.01 * np.arange(len(texts))[:, None]

    monkeypatch.setattr(D.DiverseProfileSampler, "_embed", _fake_embed)
    sampler = D.DiverseProfileSampler(profiles, seed=0, synth_frac=1.0)
    out = sampler.sample_many(4)
    assert len(out) == 4
    assert all(isinstance(p, dict) for p in out)
    texts = D.build_diverse_texts(profiles, 3, seed=1)
    assert len(texts) == 3
    assert all("Implicit user profile" in t for t in texts)
