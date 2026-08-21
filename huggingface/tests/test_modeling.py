"""CueModel encode/decode surface, guards, and sampled-user path."""

from __future__ import annotations

import numpy as np
import pytest
import torch
from cue_hf import CueModel, CueProcessor
from cue_hf.example_pool import ExamplePool, PoolExample
from tests.conftest import make_sampler_payload


def test_encode_shape_and_determinism(tiny_model, sessions):
    first = tiny_model.encode(sessions)
    assert first.shape == (2, tiny_model.config.bottleneck_dim)
    assert torch.equal(first, tiny_model.encode(sessions))


def test_encode_accepts_processor_output(tiny_model, sessions):
    normalized = CueProcessor()(sessions)
    assert tiny_model.encode(normalized).shape == (2, tiny_model.config.bottleneck_dim)


def test_encode_is_unnormalized(tiny_model, sessions):
    norms = tiny_model.encode(sessions).norm(dim=-1)
    assert not torch.allclose(norms, torch.ones_like(norms))


def test_session_preprocess_changes_the_embedding(tiny_model):
    draft = "Subject: Update\n\n" + " ".join(["body"] * 200)
    # The trailing user turn matters: assistant text only reaches the encoder as context
    # for a later user turn.
    session = [
        [
            {"role": "user", "content": "write it"},
            {"role": "assistant", "content": draft},
            {"role": "user", "content": "shorter"},
        ]
    ]
    # At init tanh(gate)=0, which zeroes the assistant-context path; open the gate so the
    # preprocess mode can actually move the embedding.
    gate = tiny_model.encoder.system_fusion.gate
    with torch.no_grad():
        gate.fill_(1.0)
    try:
        full = tiny_model.encode(session, session_preprocess="full")
        stripped = tiny_model.encode(session, session_preprocess="strip_document")
    finally:
        with torch.no_grad():
            gate.zero_()
    assert not torch.allclose(full, stripped)


def test_generate_manual_returns_one_entry_per_session(tiny_model, sessions):
    manuals = tiny_model.generate_manual(sessions=sessions, max_new_tokens=4)
    assert len(manuals) == 2
    assert all(manual is None or "commands" in manual for manual in manuals)


def test_generate_manual_accepts_embeddings(tiny_model):
    embeddings = torch.zeros(3, tiny_model.config.bottleneck_dim)
    assert len(tiny_model.generate_manual(embeddings=embeddings, max_new_tokens=4)) == 3


def test_generate_manual_requires_exactly_one_input(tiny_model, sessions):
    with pytest.raises(ValueError, match="exactly one"):
        tiny_model.generate_manual()
    with pytest.raises(ValueError, match="exactly one"):
        tiny_model.generate_manual(sessions=sessions, embeddings=torch.zeros(2, tiny_model.config.bottleneck_dim))


def test_forward_rejects_input_ids(tiny_model):
    with pytest.raises(TypeError, match="dialogue turns"):
        tiny_model(input_ids=torch.zeros(1, 4, dtype=torch.long))


def test_sample_without_sampler_explains_how_to_attach(tiny_model):
    with pytest.raises(RuntimeError, match="attach_sampler"):
        tiny_model.sample(n=1)


def test_tokenizers_are_exposed_per_component(tiny_model):
    assert tiny_model.encoder_tokenizer is tiny_model.encoder.user_tokenizer
    assert tiny_model.decoder_tokenizer is tiny_model.decoder.tokenizer


def test_save_and_reload_preserves_encode(tiny_model, sessions, tmp_path):
    before = tiny_model.encode(sessions)
    tiny_model.save_pretrained(tmp_path)
    reloaded = CueModel.from_pretrained(tmp_path)
    assert torch.equal(before, reloaded.encode(sessions))


def test_sampler_attaches_and_samples_deterministically(tiny_model, tmp_path):
    torch.save(make_sampler_payload(tiny_model.config.bottleneck_dim), tmp_path / "sampler.pt")
    tiny_model.attach_sampler(tmp_path)
    first = tiny_model.sample(n=2, seed=3)
    assert first.shape == (2, tiny_model.config.bottleneck_dim)
    assert torch.equal(first, tiny_model.sample(n=2, seed=3))
    assert not torch.equal(first, tiny_model.sample(n=2, seed=4))


def test_attaching_a_hub_sampler_fetches_one_file(tiny_model, tmp_path, monkeypatch):
    """Refined repos point sampler_id at the joint repo, whose weights must stay put."""

    torch.save(make_sampler_payload(tiny_model.config.bottleneck_dim), tmp_path / "sampler.pt")
    requested = []

    def fake_download(repo_id, filename, **kwargs):
        requested.append((repo_id, filename))
        return str(tmp_path / filename)

    monkeypatch.setattr("huggingface_hub.hf_hub_download", fake_download, raising=True)
    monkeypatch.setattr(
        "cue_hf.modeling_cue.resolve_local_dir",
        lambda *a, **k: pytest.fail("snapshotted the whole repo instead of one file"),
    )
    tiny_model.attach_sampler("org/cue")
    assert requested == [("org/cue", "sampler.pt")]
    assert tiny_model.sample(n=1, seed=0).shape == (1, tiny_model.config.bottleneck_dim)


def test_sample_user_returns_embeddings_and_manuals(tiny_model, tmp_path):
    torch.save(make_sampler_payload(tiny_model.config.bottleneck_dim), tmp_path / "sampler.pt")
    tiny_model.attach_sampler(tmp_path)
    out = tiny_model.sample_user(n=2, seed=0, max_new_tokens=4)
    assert out["embeddings"].shape == (2, tiny_model.config.bottleneck_dim)
    assert len(out["manuals"]) == 2


def test_conditional_sampling_uses_the_condition(tiny_model, tmp_path):
    torch.save(make_sampler_payload(tiny_model.config.bottleneck_dim), tmp_path / "sampler.pt")
    tiny_model.attach_sampler(tmp_path)
    dim = tiny_model.config.bottleneck_dim
    condition = np.ones((1, 2, dim), dtype=np.float32)
    conditional = tiny_model.sample(n=1, condition=condition, guidance_w=2.0, seed=0)
    unconditional = tiny_model.sample(n=1, seed=0)
    assert not torch.equal(conditional, unconditional)


def test_example_retrieval_fills_examples_from_the_pool(tiny_model, sessions):
    dim = tiny_model.config.bottleneck_dim
    pool = ExamplePool.from_rows(
        ["s0"],
        np.ones((1, dim), dtype=np.float32),
        [[PoolExample("k thanks", "human_contrast")]],
    )
    tiny_model._example_pool = pool
    manuals = tiny_model.generate_manual(sessions=sessions, max_new_tokens=4, example_retrieval=True)
    for manual in manuals:
        if manual is not None:
            assert manual["examples"] == ["k thanks"]


def test_example_pool_dim_mismatch_is_rejected(tiny_model, tmp_path, monkeypatch):
    pool = ExamplePool.from_rows(["s0"], np.ones((1, 3), dtype=np.float32), [[PoolExample("x", "style")]])
    monkeypatch.setattr(ExamplePool, "from_hub_dataset", classmethod(lambda cls, _, **kwargs: pool))
    with pytest.raises(ValueError, match="does not match model"):
        tiny_model.attach_example_pool("org/mismatched-pool")


def test_missing_pool_warns_instead_of_failing(tiny_model, sessions, capsys):
    tiny_model._example_pool = None
    manuals = tiny_model.generate_manual(sessions=sessions, max_new_tokens=4, example_retrieval=True)
    assert len(manuals) == 2
    assert "no example pool is loaded" in capsys.readouterr().err
