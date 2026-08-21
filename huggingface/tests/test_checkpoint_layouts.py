"""Loading the three checkpoint layouts cue-hf accepts."""

from __future__ import annotations

import json

import pytest
import torch
from cue_hf import CueModel
from cue_hf.checkpoint import LAYOUT_JOINT, LAYOUT_NATIVE, LAYOUT_REFINED, detect_layout
from tests.conftest import TINY_DECODER, TINY_ENCODER


def _write_joint_checkpoint(model: CueModel, directory) -> None:
    """Mirror cue_training's save_checkpoint: unified_config.json + model.pt."""

    directory.mkdir(parents=True, exist_ok=True)
    config = model.config
    (directory / "unified_config.json").write_text(
        json.dumps(
            {
                "train": {"encoder": config.encoder, "decoder": config.decoder},
                "model": {
                    "encoder": config.encoder,
                    "decoder": config.decoder,
                    "dual_decode": config.dual_decode,
                    "command_slot_decode": config.command_slot_decode,
                    "general_command_slots": config.general_command_slots,
                    "user_specific_command_slots": config.user_specific_command_slots,
                    "style_command_slots": config.style_command_slots,
                    "slot_embedding_dim": config.slot_embedding_dim,
                },
            }
        ),
        encoding="utf-8",
    )
    torch.save({"epoch": 1, "step": 10, "model": model.state_dict()}, directory / "model.pt")


def test_joint_checkpoint_loads_and_matches_encode(tiny_model, sessions, tmp_path):
    joint = tmp_path / "joint"
    _write_joint_checkpoint(tiny_model, joint)
    assert detect_layout(joint) == LAYOUT_JOINT

    loaded = CueModel.from_pretrained(joint)
    assert loaded.config.encoder["user_model_name"] == TINY_ENCODER
    assert loaded.config.decoder["model_name"] == TINY_DECODER
    assert torch.equal(tiny_model.encode(sessions), loaded.encode(sessions))


def test_refinement_checkpoint_pairs_joint_encoder_with_refined_decoder(tiny_model, sessions, tmp_path):
    joint = tmp_path / "joint"
    _write_joint_checkpoint(tiny_model, joint)
    refined = tmp_path / "refined"
    refined.mkdir()
    (refined / "refinement_config.json").write_text(
        json.dumps({"joint_training_checkpoint": str(joint)}), encoding="utf-8"
    )
    # A decoder whose persona projection is perturbed, so it is detectably different.
    decoder_state = {
        key: (value + 1.0 if key.startswith("memory.") else value)
        for key, value in tiny_model.decoder.state_dict().items()
    }
    torch.save(decoder_state, refined / "decoder_state.pt")
    assert detect_layout(refined) == LAYOUT_REFINED

    loaded = CueModel.from_pretrained(refined)
    # Encoder comes from the joint checkpoint...
    assert torch.equal(tiny_model.encode(sessions), loaded.encode(sessions))
    # ...and the decoder from the refinement directory.
    assert torch.allclose(
        loaded.decoder.memory.proj[0].bias,
        tiny_model.decoder.memory.proj[0].bias + 1.0,
    )


def test_refinement_can_point_at_a_hub_repo_for_its_encoder(tiny_model, sessions, tmp_path, monkeypatch):
    """Lets refined variants ship without duplicating the joint encoder per repo."""

    joint = tmp_path / "joint"
    _write_joint_checkpoint(tiny_model, joint)
    refined = tmp_path / "refined"
    refined.mkdir()
    (refined / "refinement_config.json").write_text(
        json.dumps({"joint_training_checkpoint": "org/cue"}), encoding="utf-8"
    )
    torch.save(tiny_model.decoder.state_dict(), refined / "decoder_state.pt")

    downloaded = []

    def fake_snapshot_download(repo_id, **kwargs):
        downloaded.append(repo_id)
        return str(joint)

    monkeypatch.setattr("huggingface_hub.snapshot_download", fake_snapshot_download, raising=True)
    loaded = CueModel.from_pretrained(refined)
    assert downloaded == ["org/cue"]
    assert torch.equal(tiny_model.encode(sessions), loaded.encode(sessions))


def test_refinement_without_joint_pointer_is_rejected(tmp_path):
    refined = tmp_path / "refined"
    refined.mkdir()
    (refined / "refinement_config.json").write_text("{}", encoding="utf-8")
    torch.save({}, refined / "decoder_state.pt")
    with pytest.raises(ValueError, match="joint training checkpoint"):
        CueModel.from_pretrained(refined)


def test_native_layout_is_detected(tiny_model, tmp_path):
    tiny_model.save_pretrained(tmp_path)
    assert detect_layout(tmp_path) == LAYOUT_NATIVE


def test_unrecognized_directory_names_the_files_it_wanted(tmp_path):
    with pytest.raises(ValueError, match="unified_config.json"):
        detect_layout(tmp_path)


def test_sampler_in_the_checkpoint_dir_is_attached_automatically(tiny_model, tmp_path):
    from tests.conftest import make_sampler_payload

    tiny_model.save_pretrained(tmp_path)
    torch.save(make_sampler_payload(tiny_model.config.bottleneck_dim), tmp_path / "sampler.pt")
    loaded = CueModel.from_pretrained(tmp_path)
    assert loaded.sampler is not None
    assert loaded.sample(n=1).shape == (1, tiny_model.config.bottleneck_dim)
