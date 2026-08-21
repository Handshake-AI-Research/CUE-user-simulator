"""Guard against drift between bundled cue-hf code and cue_training.

Skipped unless cue_training is importable (install with ``uv sync --group research``
or ``uv run --group research``). Both trees build the same tiny architecture, share
weights, and must produce the same bottleneck for the same session.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import torch
from cue_hf import CueModel
from tests.conftest import TINY_DECODER, TINY_ENCODER, tiny_config


def _import_cue_training():
    pytest.importorskip("cue_training")
    try:
        from cue_training.model import build_unified_model
    except ImportError as exc:  # pragma: no cover - environment dependent
        pytest.skip(f"cue_training not importable: {exc}")
    return build_unified_model


def build_upstream(cfg):
    """Build cue_training's model, skipping when the tiny backbones cannot be fetched."""

    build_unified_model = _import_cue_training()
    torch.manual_seed(0)
    try:
        return build_unified_model(cfg).eval()
    except Exception as exc:  # noqa: BLE001 - hub unreachable or cache miss
        pytest.skip(f"tiny backbones unavailable: {exc}")


SESSION = [
    [
        {"role": "user", "content": "hey can you fix this paragraph"},
        {"role": "assistant", "content": "Sure — here is a revised version."},
        {"role": "user", "content": "shorter pls"},
    ]
]


def test_encode_matches_cue_training():
    config = tiny_config()
    upstream_cfg = {
        "encoder": config.encoder,
        "decoder": config.decoder,
        "dual_decode": config.dual_decode,
        "command_slot_decode": config.command_slot_decode,
        "general_command_slots": config.general_command_slots,
        "user_specific_command_slots": config.user_specific_command_slots,
        "style_command_slots": config.style_command_slots,
        "slot_embedding_dim": config.slot_embedding_dim,
    }
    upstream = build_upstream(upstream_cfg)
    torch.manual_seed(0)
    model = CueModel(config, pretrained_backbones=False).eval()
    # Share weights rather than trusting two random inits to line up.
    missing, _unexpected = model.load_state_dict(upstream.state_dict(), strict=False)
    assert not missing, f"cue-hf has weights cue_training does not: {missing[:5]}"

    with torch.no_grad():
        expected = upstream.encode(SESSION)
        actual = model.encode(SESSION)
    assert torch.allclose(expected, actual, atol=1e-6), (expected - actual).abs().max()


_PREPROCESS_CASES = (
    "",
    "shorter pls",
    "Do you want it formal or casual? Should I keep the closing line?",
    "Sure — here's a tighter version.\n\n" + " ".join(["body"] * 120),
    "Here's the draft.\nSubject: Q3 update\nDear team, " + " ".join(["text"] * 90),
    " ".join(["undifferentiated prose"] * 60),
    "I can revise the draft; here is the updated version with a lighter tone.\n\n" + " ".join(["paragraph"] * 150),
)


@pytest.mark.parametrize("text", _PREPROCESS_CASES)
def test_strip_document_matches_cue_training(text):
    _import_cue_training()
    from cue_hf.session_preprocess import strip_assistant_document
    from cue_training.infer.session_preprocess import (
        strip_assistant_document as upstream,
    )

    assert strip_assistant_document(text) == upstream(text)


def test_defaults_match_the_joint_training_config():
    """Defaults claim to reproduce the training run, so read that run's config and check."""

    import json

    from cue_hf.configuration_cue import CueConfig

    _import_cue_training()
    root = Path(__file__).resolve().parents[2]
    upstream = json.loads((root / "training" / "configs" / "joint_training.json").read_text())
    config = CueConfig()
    for key in (
        "general_command_slots",
        "user_specific_command_slots",
        "style_command_slots",
        "slot_embedding_dim",
        "target_max_tokens",
        "dual_decode",
        "command_slot_decode",
    ):
        assert getattr(config, key) == upstream[key], key
    for key in ("query_prefix", "passage_prefix", "session_layers", "session_heads"):
        assert config.encoder[key] == upstream["encoder"][key], key
    for key in ("model_name", "num_persona_tokens", "insert_every", "num_heads"):
        assert config.decoder[key] == upstream["decoder"][key], key


def test_defaults_match_cue_training_build_fallbacks():
    """cue_training's build defaults are what the released checkpoint was trained with."""

    _import_cue_training()
    import inspect

    from cue_hf.configuration_cue import CueConfig
    from cue_training.encoder.model import UnifiedEncoder

    upstream_defaults = inspect.signature(UnifiedEncoder.__init__).parameters
    config = CueConfig()
    assert config.encoder["query_prefix"] == upstream_defaults["query_prefix"].default
    assert config.encoder["passage_prefix"] == upstream_defaults["passage_prefix"].default
    assert config.encoder["session_layers"] == upstream_defaults["session_layers"].default
    assert config.encoder["bottleneck_dim"] == upstream_defaults["bottleneck_dim"].default
    assert config.decoder["num_persona_tokens"] == 16
    assert (TINY_ENCODER, TINY_DECODER) != (
        config.encoder["user_model_name"],
        config.decoder["model_name"],
    )
