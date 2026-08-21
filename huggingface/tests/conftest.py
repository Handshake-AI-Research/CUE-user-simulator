"""Tiny CPU-only fixtures.

The real defaults (ModernBERT + Qwen3) are too big for tests, so these build the same
architecture around tiny cached backbones. Set ``HF_HUB_OFFLINE=1`` to force cache use.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch
from cue_hf import CueConfig, CueModel

TINY_ENCODER = "hf-internal-testing/tiny-random-bert"
TINY_DECODER = "sshleifer/tiny-gpt2"


def tiny_config(**overrides) -> CueConfig:
    kwargs = {
        "encoder": {
            "user_model_name": TINY_ENCODER,
            "system_model_name": TINY_ENCODER,
            "bottleneck_dim": 16,
            "session_dim": 16,
            "session_layers": 1,
            "session_heads": 4,
            "max_turns": 8,
            "max_length": 32,
        },
        "decoder": {
            "model_name": TINY_DECODER,
            "num_persona_tokens": 2,
            "insert_every": 1,
            "num_heads": 2,
            "dtype": "float32",
        },
        "general_command_slots": 1,
        "user_specific_command_slots": 1,
        "style_command_slots": 1,
        "decode_max_new_tokens": 8,
    }
    kwargs.update(overrides)
    return CueConfig(**kwargs)


@pytest.fixture(scope="session")
def tiny_model() -> CueModel:
    torch.manual_seed(0)
    try:
        model = CueModel(tiny_config(), pretrained_backbones=False)
    except Exception as exc:  # noqa: BLE001 - offline cache miss
        pytest.skip(f"tiny backbones unavailable: {exc}")
    return model.eval()


@pytest.fixture(autouse=True)
def _detach_side_artifacts(request) -> None:
    """The tiny model is session-scoped, so tests must not leak sampler/pool state."""

    yield
    model = request.node.funcargs.get("tiny_model") if hasattr(request.node, "funcargs") else None
    if model is not None:
        model._sampler = None
        model._example_pool = None


@pytest.fixture
def sessions() -> list[list[dict[str, str]]]:
    return [
        [
            {"role": "user", "content": "hey can you fix this paragraph"},
            {"role": "assistant", "content": "Sure — here is a revised version."},
            {"role": "user", "content": "shorter pls"},
        ],
        [{"role": "user", "content": "what time does the store close"}],
    ]


def make_sampler_payload(dim: int, *, width: int = 8, depth: int = 1) -> dict:
    """A structurally valid ``sampler.pt`` payload with random weights."""

    from cue_hf.sampler.diffusion import CosineNoiseSchedule, DiffusionSamplerModel

    torch.manual_seed(0)
    model = DiffusionSamplerModel(dim=dim, width=width, depth=depth, conditioner_heads=1)
    schedule = CosineNoiseSchedule(timesteps=20)
    return {
        "model": model.state_dict(),
        "schedule": schedule.state_dict(),
        "mu": np.zeros(dim, dtype=np.float32),
        "sigma": np.ones(dim, dtype=np.float32),
        "config": {
            "dim": dim,
            "width": width,
            "depth": depth,
            "conditioner_heads": 1,
            "ddim_steps": 4,
            "guidance_w": 1.5,
        },
    }
