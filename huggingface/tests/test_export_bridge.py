"""`cue-train export-cue-hf` must produce a byte-faithful cue-hf model.

Skipped unless cue_training is importable (see tests/test_encode_parity.py).
"""

from __future__ import annotations

import json

import pytest
import torch
from cue_hf import CueModel
from tests.conftest import tiny_config
from tests.test_encode_parity import SESSION, _import_cue_training, build_upstream


def _export_cue_hf():
    """Import the exporter only after cue_training is available (else skip, not error)."""

    _import_cue_training()
    from cue_training.infer.export_cue_hf import export_cue_hf

    return export_cue_hf


def _tiny_joint_checkpoint(tmp_path):
    _import_cue_training()
    from cue_training.utils.checkpoint import save_checkpoint

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
    checkpoint = tmp_path / "joint"
    save_checkpoint(
        checkpoint,
        config={"train": upstream_cfg, "model": upstream.config_dict()},
        model=upstream,
    )
    return upstream, checkpoint


def test_export_preserves_encode_and_lm_head(tmp_path):
    export_cue_hf = _export_cue_hf()

    upstream, checkpoint = _tiny_joint_checkpoint(tmp_path)
    exported = export_cue_hf(checkpoint, tmp_path / "cue-hf")
    model = CueModel.from_pretrained(exported)

    with torch.no_grad():
        assert torch.allclose(upstream.encode(SESSION), model.encode(SESSION), atol=1e-6)
    # The tied output embedding must come back as the trained one, not a fresh init.
    assert torch.equal(
        upstream.decoder.lm.get_output_embeddings().weight,
        model.decoder.lm.get_output_embeddings().weight,
    )


def test_export_of_a_refinement_checkpoint_carries_the_refined_decoder(tmp_path):
    """Each refined variant exports to a standalone repo, so the refined decoder must win."""

    export_cue_hf = _export_cue_hf()

    upstream, joint = _tiny_joint_checkpoint(tmp_path)
    refined_dir = tmp_path / "refined"
    refined_dir.mkdir()
    (refined_dir / "refinement_config.json").write_text(
        json.dumps({"joint_training_checkpoint": str(joint)}), encoding="utf-8"
    )
    refined_decoder = {
        key: value + 0.05 if value.is_floating_point() else value
        for key, value in upstream.decoder.state_dict().items()
    }
    torch.save(refined_decoder, refined_dir / "decoder_state.pt")

    model = CueModel.from_pretrained(export_cue_hf(refined_dir, tmp_path / "cue-hf"))
    with torch.no_grad():
        # Encoder still comes from the joint checkpoint.
        assert torch.allclose(upstream.encode(SESSION), model.encode(SESSION), atol=1e-6)
    exported = model.decoder.state_dict()
    key = next(k for k, v in refined_decoder.items() if v.is_floating_point())
    assert torch.allclose(exported[key], refined_decoder[key], atol=1e-6)


def test_example_pool_dataset_round_trips_into_retrieval(tmp_path, monkeypatch):
    _import_cue_training()
    datasets = pytest.importorskip("datasets", reason="needs the retrieval extra")
    import numpy as np

    # Keep the datasets builder cache out of the user's home directory.
    monkeypatch.setattr(datasets.config, "HF_DATASETS_CACHE", tmp_path / "hf-cache")

    from cue_hf.example_pool import ExamplePool
    from cue_training.data.example_pool import ExamplePool as UpstreamPool
    from cue_training.data.example_pool import PoolExample as UpstreamExample
    from cue_training.infer.export_cue_hf import export_example_pool_dataset
    from cue_training.sampler.faiss_index import EmbeddingIndex

    vectors = np.eye(2, dtype=np.float32)
    index = EmbeddingIndex(2)
    index.add(vectors)
    checkpoint = tmp_path / "ckpt"
    (checkpoint / "example_pool").mkdir(parents=True)
    UpstreamPool(
        ["s0", "s1"],
        vectors,
        [[UpstreamExample("k thanks", "human_contrast")], [UpstreamExample("sure", "style")]],
        index,
        "fingerprint-1",
    ).save(checkpoint / "example_pool")

    export_example_pool_dataset(checkpoint, output=tmp_path / "pool-dataset")
    pool = ExamplePool.from_hub_dataset(str(tmp_path / "pool-dataset"))
    assert pool.session_ids == ["s0", "s1"]
    assert pool.checkpoint_fingerprint == "fingerprint-1"
    assert pool.retrieve([1.0, 0.0], n_general=0, n_specific=1) == ["k thanks"]


def test_exported_config_records_session_shape(tmp_path):
    export_cue_hf = _export_cue_hf()

    _, checkpoint = _tiny_joint_checkpoint(tmp_path)
    model = CueModel.from_pretrained(export_cue_hf(checkpoint, tmp_path / "cue-hf"))
    expected = tiny_config()
    assert model.config.encoder["session_layers"] == expected.encoder["session_layers"]
    assert model.config.encoder["session_heads"] == expected.encoder["session_heads"]
    assert model.config.decoder["dtype"] == expected.decoder["dtype"]
