"""Slimming a sampler.pt must not change what it samples."""

from __future__ import annotations

import torch
from cue_hf.sampler.sample import (
    INFERENCE_KEYS,
    load_sampler,
    slim_sampler_file,
    slim_sampler_payload,
)
from tests.conftest import make_sampler_payload

DIM = 16


def _training_payload():
    """A payload shaped like a training checkpoint: raw weights, an EMA copy, optimizer state."""

    payload = make_sampler_payload(DIM)
    payload["ema"] = {
        "shadow": {
            key: (value + 0.25 if value.is_floating_point() else value) for key, value in payload["model"].items()
        },
        "num_updates": 1000,
    }
    payload["optimizer"] = {
        "state": {i: {"exp_avg": torch.ones(DIM)} for i in range(3)},
        "param_groups": [{"lr": 1e-4}],
    }
    payload["scheduler"] = {"last_epoch": 7}
    payload["layernorm_weight"] = torch.ones(DIM).tolist()
    payload["layernorm_bias"] = torch.zeros(DIM).tolist()
    return payload


def _a_float_key(state):
    return next(key for key, value in state.items() if value.is_floating_point())


def test_slim_drops_training_state(tmp_path):
    payload = _training_payload()
    slim = slim_sampler_payload(payload)
    assert "optimizer" not in slim
    assert "scheduler" not in slim
    assert "ema" not in slim
    assert set(slim) <= {"model", *INFERENCE_KEYS}
    # The EMA weights are what inference uses, so they must be the copy that survives.
    key = _a_float_key(payload["model"])
    assert torch.equal(slim["model"][key], payload["ema"]["shadow"][key])
    assert not torch.equal(slim["model"][key], payload["model"][key])


def test_slim_file_samples_identically(tmp_path):
    source = tmp_path / "sampler.pt"
    torch.save(_training_payload(), source)
    destination = slim_sampler_file(source, tmp_path / "slim.pt")
    assert destination.stat().st_size < source.stat().st_size

    reference = load_sampler(source).sample(n=2, seed=0)
    slimmed = load_sampler(destination).sample(n=2, seed=0)
    assert torch.allclose(reference, slimmed, atol=1e-6)
    # Guard against silently shipping the non-EMA weights, which sample differently.
    raw = load_sampler(source, use_ema=False).sample(n=2, seed=0)
    assert not torch.allclose(raw, slimmed, atol=1e-3)


def test_slim_falls_back_to_model_when_there_is_no_ema(tmp_path):
    payload = make_sampler_payload(DIM)
    slim = slim_sampler_payload(payload)
    key = _a_float_key(payload["model"])
    assert torch.equal(slim["model"][key], payload["model"][key])
