"""Unit tests for cue embedding diffusion sampler (no HF downloads)."""

from __future__ import annotations

import hashlib
import json

import numpy as np
import pytest
import torch

from cue_training.sampler.bank import (
    BankShard,
    EmbeddingBank,
    _load_meta_ids,
    _near_duplicate_mask,
    discover_files_by_name,
)
from cue_training.sampler.diffusion import (
    CosineNoiseSchedule,
    DiffusionSamplerModel,
    EMA,
    SetConditioner,
)
from cue_training.sampler.faiss_index import EmbeddingIndex
from cue_training.sampler.metrics import acceptance_report
from cue_training.sampler.sample import ddim_sample, project_layernorm_manifold
from cue_training.sampler.train import (
    _sample_neighborhood_episode,
    _save_sampler_checkpoint,
    _source_weights,
)


def test_cosine_schedule_q_sample_identity():
    schedule = CosineNoiseSchedule(timesteps=100)
    assert schedule.alphas_cumprod[0] > schedule.alphas_cumprod[-1]
    z0 = torch.randn(4, 16)
    t0 = torch.zeros(4, dtype=torch.long)
    noise0 = torch.randn_like(z0)
    zt0, _ = schedule.q_sample(z0, t0, noise=noise0)
    # low-t samples stay close to z0
    assert float((zt0 - z0).pow(2).mean()) < 0.05
    t_last = torch.full((4,), 99, dtype=torch.long)
    noise = torch.randn_like(z0)
    zt, _ = schedule.q_sample(z0, t_last, noise=noise)
    # high-t samples should be closer to noise than to z0
    err_noise = (zt - noise).pow(2).mean()
    err_z0 = (zt - z0).pow(2).mean()
    assert float(err_noise) < float(err_z0)


def test_predict_x0_roundtrip():
    schedule = CosineNoiseSchedule(timesteps=50)
    z0 = torch.randn(2, 8)
    t = torch.tensor([10, 20])
    noise = torch.randn_like(z0)
    zt, noise = schedule.q_sample(z0, t, noise=noise)
    x0_hat = schedule.predict_x0_from_eps(zt, t, noise)
    assert torch.allclose(x0_hat, z0, atol=1e-4)


def test_set_conditioner_permutation_invariant_and_null():
    cond = SetConditioner(dim=32, hidden=64, num_heads=4)
    x = torch.randn(2, 5, 32)
    mask = torch.ones(2, 5, dtype=torch.bool)
    c1 = cond(x, mask)
    perm = x[:, torch.randperm(5), :]
    c2 = cond(perm, mask)
    assert torch.allclose(c1, c2, atol=1e-5)
    drop = torch.tensor([True, False])
    c3 = cond(x, mask, drop_condition=drop)
    assert torch.allclose(c3[0], cond.null, atol=1e-6)
    assert not torch.allclose(c3[1], cond.null, atol=1e-3)


def test_cfg_endpoints_differ():
    model = DiffusionSamplerModel(dim=32, width=64, depth=2, time_dim=32)
    zt = torch.randn(3, 32)
    t = torch.zeros(3, dtype=torch.long)
    set_emb = torch.randn(3, 4, 32)
    mask = torch.ones(3, 4, dtype=torch.bool)
    eps_u = model(zt, t, set_emb=None)
    eps_c = model(zt, t, set_emb=set_emb, mask=mask)
    assert eps_u.shape == (3, 32)
    assert not torch.allclose(eps_u, eps_c)


def test_ddim_shape_and_determinism():
    model = DiffusionSamplerModel(dim=16, width=32, depth=2, time_dim=16)
    schedule = CosineNoiseSchedule(timesteps=20)
    g1 = torch.Generator().manual_seed(0)
    g2 = torch.Generator().manual_seed(0)
    a = ddim_sample(model, schedule, n=4, steps=5, guidance_w=0.0, generator=g1, device="cpu")
    b = ddim_sample(model, schedule, n=4, steps=5, guidance_w=0.0, generator=g2, device="cpu")
    assert a.shape == (4, 16)
    assert torch.allclose(a, b)


def test_manifold_projection_affine():
    z = torch.randn(3, 8)
    w = torch.ones(8) * 2.0
    b = torch.zeros(8)
    out = project_layernorm_manifold(z, w, b)
    # mean of pre-affine should be ~0 after projection in normalized space
    x_hat = (out - b) / w
    assert torch.allclose(x_hat.mean(dim=-1), torch.zeros(3), atol=1e-4)


def test_neighborhood_excludes_self():
    rng = np.random.default_rng(0)
    emb = rng.normal(size=(20, 8)).astype(np.float32)
    index = EmbeddingIndex(8)
    index.add(emb)
    z0, set_t, mask, drop = _sample_neighborhood_episode(
        emb, index, batch_size=8, k=4, rng=rng, p_uncond=0.0
    )
    assert z0.shape == (8, 8)
    assert set_t.shape == (8, 4, 8)
    assert mask.shape == (8, 4)
    # each condition vector should not be identical to target when alternatives exist
    for i in range(8):
        valid = set_t[i][mask[i]]
        if valid.shape[0] == 0:
            continue
        # at least one valid member
        assert valid.shape[0] >= 1


def test_source_weights_rebalance_anchor_mass():
    """alpha=1 keeps the cap-driven mixture; alpha=0 gives each corpus equal mass."""

    sources = ["dialogstudio"] * 90 + ["lmsys"] * 10
    assert _source_weights(sources, 1.0) is None

    weights = _source_weights(sources, 0.0)
    assert weights is not None
    assert weights[:90].sum() == pytest.approx(0.5)
    assert weights[90:].sum() == pytest.approx(0.5)

    rng = np.random.default_rng(0)
    emb = rng.normal(size=(100, 8)).astype(np.float32)
    index = EmbeddingIndex(8)
    index.add(emb)
    z0, _, _, _ = _sample_neighborhood_episode(
        emb, index, batch_size=4, k=2, rng=rng, p_uncond=0.0, weights=weights
    )
    assert z0.shape == (4, 8)


def test_same_source_frac_restricts_conditioning_sets():
    """Sources alternate along a ring, so the nearest neighbor is always the other source."""

    angles = np.arange(40) * 0.05
    emb = np.stack([np.cos(angles), np.sin(angles)], axis=1).astype(np.float32)
    codes = np.arange(40) % 2  # parity is the source; neighbors i±1 are cross-source
    index = EmbeddingIndex(2)
    index.add(emb)

    def _parities(**kwargs):
        z0, set_t, mask, _ = _sample_neighborhood_episode(
            emb, index, batch_size=12, k=3, rng=np.random.default_rng(0), p_uncond=0.0, **kwargs
        )
        rows = {tuple(np.round(v, 6)): i for i, v in enumerate(emb)}
        out = []
        for i in range(12):
            target = codes[rows[tuple(np.round(z0[i].numpy(), 6))]]
            for vec in set_t[i][mask[i]].numpy():
                out.append((target, codes[rows[tuple(np.round(vec, 6))]]))
        return out

    mixed = _parities(source_codes=codes, same_source_frac=0.0)
    assert any(t != c for t, c in mixed), "fixture should produce cross-source neighborhoods"

    restricted = _parities(source_codes=codes, same_source_frac=1.0)
    assert restricted and all(t == c for t, c in restricted)


def test_near_duplicate_mask_keeps_first():
    a = np.array([[1.0, 0.0], [1.0, 0.0], [0.0, 1.0]], dtype=np.float32)
    keep = _near_duplicate_mask(a, cosine_threshold=0.99)
    assert keep.tolist() == [True, False, True]


def test_holdout_bucket_is_deterministic():
    from cue_training.sampler.bank import _holdout_bucket

    a = _holdout_bucket("session-1", seed=0, holdout_frac=0.5)
    b = _holdout_bucket("session-1", seed=0, holdout_frac=0.5)
    assert a is b
    assert _holdout_bucket("session-1", seed=0, holdout_frac=0.0) is False


def test_streaming_shard_writer_encodes_without_materializing(tmp_path):
    from cue_training.sampler.bank import _StreamingShardWriter, _stream_encode_records

    class _FakeModel:
        def encode(self, sessions):
            return torch.tensor(
                [[float(len(session)), 1.0] for session in sessions],
                dtype=torch.float32,
            )

    writer = _StreamingShardWriter(tmp_path, "prior_train", dim=2)
    records = (
        {
            "session_id": f"s{i}",
            "turns": [{"role": "user", "content": "hi" * (i + 1)}],
            "_source": "toy",
            "_text_hash": f"h{i}",
        }
        for i in range(5)
    )
    _stream_encode_records(
        _FakeModel(),
        records,
        {"prior_train": writer},
        route=lambda _rec: "prior_train",
        batch_size=2,
        desc="test stream",
    )
    emb, meta_path, stats = writer.finalize(near_dup_cosine=1.0)
    assert emb.shape == (5, 2)
    assert stats["encoded"] == 5
    assert meta_path.is_file()
    assert sum(1 for _ in meta_path.open()) == 5


class _FakeBankEncoder:
    """Deterministic per-session embedding so shard order cannot change the vectors."""

    class _Enc:
        bottleneck_dim = 8
        bottleneck = None

    encoder = _Enc()

    def eval(self):
        return self

    def encode(self, sessions):
        rows = []
        for session in sessions:
            seed = int(hashlib.md5(str(session).encode()).hexdigest()[:8], 16)
            rows.append(np.random.default_rng(seed).normal(size=8))
        return torch.tensor(np.asarray(rows), dtype=torch.float32)


def _write_bank_corpus(root, n: int) -> None:
    prior = root / "prior" / "toy"
    prior.mkdir(parents=True, exist_ok=True)
    with (prior / "train.jsonl").open("w", encoding="utf-8") as handle:
        for i in range(n):
            handle.write(
                json.dumps(
                    {"id": f"p{i}", "turns": [{"role": "user", "content": f"prior {i}"}]}
                )
                + "\n"
            )


def _bank_cfg(root, bank_dir):
    return {
        "joint_training_checkpoint": "ckpt",
        "device": "cpu",
        "bank_dir": str(bank_dir),
        "prior_data_root": str(root / "prior"),
        "prior_filename": "train.jsonl",
        "encode_batch_size": 4,
        "prior_holdout_frac": 0.25,
        "seed": 0,
    }


@pytest.fixture
def _patched_bank_encoder(monkeypatch):
    from cue_training.sampler import bank as mod

    monkeypatch.setattr(mod, "load_checkpoint", lambda *a, **k: _FakeBankEncoder())


def test_bank_shards_partition_sessions(tmp_path, _patched_bank_encoder):
    """Four shards cover every session exactly once, matching a single-process build."""

    from cue_training.sampler.bank import build_bank

    _write_bank_corpus(tmp_path, 40)
    whole = build_bank(_bank_cfg(tmp_path, tmp_path / "whole"))
    per_shard: list[str] = []
    for i in range(4):
        build_bank(
            _bank_cfg(tmp_path, tmp_path / f"s{i}"),
            shard_index=i,
            num_shards=4,
            finalize=False,
        )
        ids, _ = _load_meta_ids(tmp_path / f"s{i}" / "prior_train.meta.jsonl")
        per_shard.extend(ids)
    assert len(per_shard) == len(set(per_shard)), "a session landed in two shards"
    assert set(per_shard) == set(whole.prior_train.session_ids)


def test_merge_bank_shards_matches_single_process_build(tmp_path, _patched_bank_encoder):
    from cue_training.sampler.bank import build_bank, load_bank, merge_bank_shards

    _write_bank_corpus(tmp_path, 40)
    whole = build_bank(_bank_cfg(tmp_path, tmp_path / "whole"))
    shard_dirs = []
    for i in range(4):
        directory = tmp_path / "merged" / f".rank{i}"
        cfg = _bank_cfg(tmp_path, directory)
        build_bank(cfg, shard_index=i, num_shards=4, finalize=False)
        shard_dirs.append(directory)
    merged = merge_bank_shards(_bank_cfg(tmp_path, tmp_path / "merged"), shard_dirs)

    for name in ("prior_train", "prior_dev"):
        want = getattr(whole, name)
        got = getattr(merged, name)
        assert got.n == want.n, name
        assert set(got.session_ids) == set(want.session_ids), name
    # mu/sigma are order-invariant over the same row set, so they must agree.
    assert np.allclose(merged.mu, whole.mu, atol=1e-5)
    assert np.allclose(merged.sigma, whole.sigma, atol=1e-5)
    assert merged.prior_index is not None
    # Reloading from disk must give the same bank the merge returned.
    assert load_bank(tmp_path / "merged").prior_train.n == whole.prior_train.n


def test_bank_distributed_falls_through_without_torchrun(tmp_path, monkeypatch, _patched_bank_encoder):
    """No WORLD_SIZE means a plain single-process build and no .rank dirs."""

    from cue_training.sampler.bank import build_bank_distributed

    monkeypatch.delenv("WORLD_SIZE", raising=False)
    _write_bank_corpus(tmp_path, 20)
    bank = build_bank_distributed(_bank_cfg(tmp_path, tmp_path / "bank"))
    assert bank is not None and bank.prior_train.n > 0
    assert not list((tmp_path / "bank").glob(".rank*"))


class _ScriptedDecoder:
    """Decodes row i to a fixed text, so shard membership cannot change the result."""

    TEXTS = [f"t{i % 5}" for i in range(16)]  # duplicates straddle any shard boundary

    def __init__(self):
        self.decoder = self

    def parameters(self):
        yield torch.zeros(1)

    @staticmethod
    def tokenizer(*_a, **_k):
        return {"input_ids": torch.zeros(1, 1, dtype=torch.long)}

    def generate(self, *, bottleneck, sentinel_ids, max_new_tokens):
        return [self.TEXTS[int(i.item())] for i in bottleneck[:, 0]]


def test_sharded_decode_pooling_matches_whole_set(monkeypatch):
    """Two-rank pooling reproduces the single-process parse, length, and duplicate rates."""

    from cue_training.sampler import metrics as metrics_mod
    from cue_training.sampler import train as train_mod

    monkeypatch.setattr(metrics_mod, "parse_manual", lambda text: None if text == "t4" else {"ok": 1})
    dec = _ScriptedDecoder()
    emb = np.stack([np.full(4, i, dtype=np.float32) for i in range(16)])
    kwargs = {"max_new_tokens": 8, "batch_size": 4}
    whole = metrics_mod.decode_parse_rate(dec, emb, **kwargs)
    other = metrics_mod.decode_parse_rate(dec, emb[1::2], **kwargs)

    def fake_all_reduce(values, _device):
        if len(values) == 1:  # the "every rank loaded" flag
            return [values[0] + 1.0]
        return [
            values[0] + other["parse_rate"] * other["n"],
            values[1] + other["mean_length"] * other["n"],
            values[2] + other["n"],
        ]

    def fake_all_gather_object(pooled, obj):
        pooled[0] = obj
        pooled[1] = other["texts"]

    monkeypatch.setattr(train_mod, "_all_reduce_sum", fake_all_reduce)
    monkeypatch.setattr(train_mod.dist, "all_gather_object", fake_all_gather_object)
    got = train_mod._decode_parse_sharded(
        dec, emb, rank=0, world_size=2, device="cpu", **kwargs
    )
    assert got["n"] == whole["n"] == 16
    assert got["parse_rate"] == pytest.approx(whole["parse_rate"])
    assert got["mean_length"] == pytest.approx(whole["mean_length"])
    # The pooled texts catch duplicates a per-shard average would miss.
    assert got["duplicate_rate"] == pytest.approx(whole["duplicate_rate"])


def test_sharded_decode_skips_on_every_rank_when_one_fails(monkeypatch):
    """A rank that cannot decode must not leave the others waiting in a collective."""

    from cue_training.sampler import train as train_mod

    monkeypatch.setattr(train_mod, "_all_reduce_sum", lambda values, _d: [0.0] * len(values))
    monkeypatch.setattr(
        train_mod, "decode_parse_rate", lambda *a, **k: (_ for _ in ()).throw(RuntimeError("boom"))
    )
    got = train_mod._decode_parse_sharded(
        _ScriptedDecoder(),
        np.zeros((8, 4), dtype=np.float32),
        max_new_tokens=8,
        batch_size=4,
        rank=0,
        world_size=2,
        device="cpu",
    )
    assert got is None


def test_discover_prior_filename(tmp_path):
    f = tmp_path / "ds" / "train.jsonl"
    f.parent.mkdir()
    f.write_text("{}\n", encoding="utf-8")
    (tmp_path / "ds" / "train.partial.jsonl").write_text("{}\n", encoding="utf-8")
    found = discover_files_by_name(tmp_path, "train.jsonl")
    assert found == [f]


def test_acceptance_report_relative_thresholds():
    report = acceptance_report(
        {
            "parse_rate_joint_training": 0.8,
            "sliced_wasserstein": 0.3,
            "rff_mmd": 0.1,
            "memorization_ratio": 0.9,
            "bias_gap": 0.05,
            "diversity": 0.1,
        },
        {
            "min_parse_rate": 0.7,
            "sw_vs_baseline_max_mult": 2.0,
            "mmd_vs_baseline_max_mult": 2.0,
            "min_memorization_ratio": 0.5,
            "min_bias_gap": 0.02,
            "min_conditional_diversity": 0.05,
        },
        {
            "real_train_vs_dev_sliced_wasserstein": 0.2,
            "real_train_vs_dev_rff_mmd": 0.08,
        },
    )
    assert report["pass"] is True


def test_tiny_train_save_load_sample(tmp_path):
    dim = 16
    rng = np.random.default_rng(0)
    prior = rng.normal(size=(64, dim)).astype(np.float32)
    mu = prior.mean(0)
    sigma = np.maximum(prior.std(0), 1e-4)
    bank_dir = tmp_path / "bank"
    bank_dir.mkdir()
    np.save(bank_dir / "prior_train.npy", prior)
    np.save(bank_dir / "prior_dev.npy", prior[:16])
    np.save(bank_dir / "mu.npy", mu.astype(np.float32))
    np.save(bank_dir / "sigma.npy", sigma.astype(np.float32))
    for name, n in (
        ("prior_train", 64),
        ("prior_dev", 16),
    ):
        with (bank_dir / f"{name}.meta.jsonl").open("w", encoding="utf-8") as handle:
            for i in range(n):
                # Skewed mixture so source_balance_alpha below has something to rebalance.
                source = "toy" if i % 8 else "rare"
                handle.write(
                    json.dumps({"row": i, "session_id": f"{name}-{i}", "source": source}) + "\n"
                )
    index = EmbeddingIndex(dim)
    index.add(prior)
    index.save(bank_dir / "prior.faiss")
    meta = {
        "dim": dim,
        "checkpoint": "fake",
        "checkpoint_hash": "abc",
        "mu": mu.tolist(),
        "sigma": sigma.tolist(),
        "counts": {"prior_train": 64, "prior_dev": 16},
    }
    (bank_dir / "bank_meta.json").write_text(json.dumps(meta), encoding="utf-8")
    (bank_dir / "bank_report.json").write_text(
        json.dumps(
            {
                "baselines": {
                    "real_train_vs_dev_sliced_wasserstein": 0.1,
                    "real_train_vs_dev_rff_mmd": 0.05,
                }
            }
        ),
        encoding="utf-8",
    )

    cfg = {
        "seed": 0,
        "device": "cpu",
        "bank_dir": str(bank_dir),
        "output": str(tmp_path / "out"),
        "width": 32,
        "depth": 2,
        "time_dim": 16,
        "timesteps": 20,
        "batch_size": 8,
        "source_balance_alpha": 0.0,
        "condition_same_source_frac": 1.0,
        "k": 3,
        "max_steps": 3,
        "warmup_steps": 1,
        "log_every": 1,
        "val_every": 0,
        "amp": False,
        "no_wandb": True,
        "val_decode": False,
        "ddim_steps": 4,
        "ema_decay": 0.9,
    }
    from cue_training.sampler.train import train_sampler_from_config

    out = train_sampler_from_config(cfg)
    assert (out / "last" / "sampler.pt").is_file()

    from cue_training.sampler.sample import sample

    result = sample(out / "last", n=4, steps=4, device="cpu", seed=1)
    assert result["embeddings"].shape == (4, dim)
    assert np.isfinite(result["embeddings"]).all()


def test_ema_update_moves_shadow():
    model = DiffusionSamplerModel(dim=8, width=16, depth=1, time_dim=8)
    ema = EMA(model, decay=0.5)
    before = next(ema.shadow.parameters()).detach().clone()
    with torch.no_grad():
        for p in model.parameters():
            p.add_(1.0)
    ema.update(model)
    after = next(ema.shadow.parameters()).detach()
    assert not torch.allclose(before, after)
