"""Tests for Wegmann style sidecar, style loss, example pool, and retrieval metrics."""

from __future__ import annotations

import inspect
import json
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import torch

from cue_training.data.example_pool import (
    ExamplePool,
    PoolExample,
    build_example_pool,
    eval_validation_retrieval,
    examples_from_manual,
)
from cue_training.data.schema import validate_manual
from cue_training.data.wegmann_style import (
    batch_style_overlap,
    encode_session_style,
    encode_session_styles,
)
from cue_training.cli import _visible_cuda_devices
from cue_training.model import command_overlap_infonce
from cue_training.sampler.faiss_index import EmbeddingIndex
from cue_training.training.train import _encode_sessions_batched


class _FakeStyleEncoder:
    def __init__(self) -> None:
        self.calls: list[list[str]] = []

    def encode(self, texts: list[str]) -> np.ndarray:
        self.calls.append(list(texts))
        out = []
        for text in texts:
            h = hash(text.casefold()) % 997
            vec = np.array([float(h % 17), float((h // 17) % 17), float((h // 289) % 17)], dtype=np.float32)
            out.append(vec)
        return np.stack(out, axis=0)


def test_example_pool_build_defaults_to_batch_256():
    assert inspect.signature(build_example_pool).parameters["batch_size"].default == 256


class _PoolBuildModel:
    class _Encoder:
        bottleneck_dim = 2

    encoder = _Encoder()

    def eval(self):
        return self

    def encode(self, sessions):
        return torch.tensor([[float(len(s)), 1.0] for s in sessions])


def _write_pool_split(root, n: int) -> None:
    split = root / "toy"
    split.mkdir(parents=True, exist_ok=True)
    with (split / "train.jsonl").open("w", encoding="utf-8") as handle:
        for i in range(n):
            handle.write(
                json.dumps(
                    {
                        "id": f"s{i}",
                        "turns": [{"role": "user", "content": f"hello {i}"}],
                        "persona_manual": {
                            "commands": [
                                {
                                    "text": f"Keep replies short {i}.",
                                    "kind": "sim_contrast",
                                    "examples": [f"sure {i}"],
                                }
                            ]
                        },
                    }
                )
                + "\n"
            )


@pytest.fixture
def _patched_pool_checkpoint(monkeypatch):
    from cue_training.data import example_pool as mod

    monkeypatch.setattr(mod, "load_checkpoint", lambda *a, **k: _PoolBuildModel())


def test_example_pool_shards_partition_sessions(tmp_path, _patched_pool_checkpoint):
    """Every session lands in exactly one shard, and 4 shards cover the whole split."""

    _write_pool_split(tmp_path / "data", 40)
    whole = build_example_pool(
        "ckpt", tmp_path / "data", tmp_path / "whole", batch_size=8
    )
    shard_ids = []
    for i in range(4):
        pool = build_example_pool(
            "ckpt", tmp_path / "data", tmp_path / f"shard{i}",
            batch_size=8, shard_index=i, num_shards=4,
        )
        shard_ids.append(set(pool.session_ids))
    union = set().union(*shard_ids)
    assert union == set(whole.session_ids)
    assert sum(len(s) for s in shard_ids) == len(union)  # disjoint


def test_merge_example_pools_matches_single_process_build(tmp_path, _patched_pool_checkpoint):
    from cue_training.data.example_pool import ExamplePool, merge_example_pools

    _write_pool_split(tmp_path / "data", 40)
    whole = build_example_pool(
        "ckpt", tmp_path / "data", tmp_path / "whole", batch_size=8
    )
    shards = []
    for i in range(4):
        build_example_pool(
            "ckpt", tmp_path / "data", tmp_path / f"shard{i}",
            batch_size=8, shard_index=i, num_shards=4,
        )
        shards.append(tmp_path / f"shard{i}")
    merged = merge_example_pools(shards, tmp_path / "merged")

    assert set(merged.session_ids) == set(whole.session_ids)
    assert merged.cue_vectors.shape == whole.cue_vectors.shape
    # Reloadable, and the rebuilt index actually retrieves.
    reloaded = ExamplePool.load(tmp_path / "merged")
    assert reloaded.ntotal == whole.ntotal
    assert reloaded.retrieve([1.0, 1.0], k_sessions=4)


def test_distributed_build_falls_through_without_torchrun(tmp_path, monkeypatch, _patched_pool_checkpoint):
    """No WORLD_SIZE means plain single-process build; no process group is created."""

    from cue_training.data.example_pool import build_example_pool_distributed

    monkeypatch.delenv("WORLD_SIZE", raising=False)
    _write_pool_split(tmp_path / "data", 12)
    pool = build_example_pool_distributed("ckpt", tmp_path / "data", tmp_path / "out")
    assert pool is not None
    assert pool.ntotal == 12
    assert (tmp_path / "out" / "meta.jsonl").is_file()
    assert not list(tmp_path.glob("out.shard*"))


def test_merge_example_pools_rejects_mixed_checkpoints(tmp_path, monkeypatch):
    from cue_training.data import example_pool as mod
    from cue_training.data.example_pool import merge_example_pools

    _write_pool_split(tmp_path / "data", 8)
    monkeypatch.setattr(mod, "load_checkpoint", lambda *a, **k: _PoolBuildModel())
    build_example_pool("ckpt_a", tmp_path / "data", tmp_path / "a", num_shards=2, shard_index=0)
    build_example_pool("ckpt_b", tmp_path / "data", tmp_path / "b", num_shards=2, shard_index=1)
    with pytest.raises(ValueError, match="different checkpoints"):
        merge_example_pools([tmp_path / "a", tmp_path / "b"], tmp_path / "merged")


class _FakeRetrievalModel:
    class _Encoder:
        bottleneck_dim = 2

    encoder = _Encoder()

    def __init__(self) -> None:
        self.batch_sizes: list[int] = []

    def encode(self, sessions):
        self.batch_sizes.append(len(sessions))
        return torch.tensor(
            [[float(session[0]["content"]), 1.0] for session in sessions]
        )


def test_retrieval_encoding_uses_configured_large_batches():
    model = _FakeRetrievalModel()
    records = [
        {"turns": [{"role": "user", "content": str(i)}]} for i in range(10)
    ]
    vectors = _encode_sessions_batched(model, records, batch_size=4)
    assert model.batch_sizes == [4, 4, 2]
    assert vectors.shape == (10, 2)


def test_encode_session_style_mean_pools_turns():
    encoder = _FakeStyleEncoder()
    turns = [
        {"role": "user", "content": "book a flight"},
        {"role": "assistant", "content": "where to?"},
        {"role": "user", "content": "nyc please"},
    ]
    style = encode_session_style(turns, encoder)
    assert style is not None
    assert style.shape == (3,)
    assert abs(float(np.linalg.norm(style)) - 1.0) < 1e-5


def test_encode_session_styles_batches_turns_across_sessions():
    encoder = _FakeStyleEncoder()
    sessions = [
        [{"role": "user", "content": "first"}, {"role": "user", "content": "second"}],
        [{"role": "assistant", "content": "ignored"}, {"role": "user", "content": "third"}],
        [{"role": "assistant", "content": "no user turns"}],
    ]
    styles = encode_session_styles(sessions, encoder)
    assert encoder.calls == [["first", "second", "third"]]
    assert styles[0] is not None
    assert styles[1] is not None
    assert styles[2] is None


def test_visible_cuda_devices_respects_mask_and_requested_count(monkeypatch):
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "2,5,7,9")
    assert _visible_cuda_devices() == ["2", "5", "7", "9"]
    assert _visible_cuda_devices(2) == ["2", "5"]
    with pytest.raises(ValueError, match="only 4"):
        _visible_cuda_devices(5)


def test_batch_style_overlap_is_symmetric_on_diagonal():
    style_map = {
        "a": np.array([1.0, 0.0, 0.0], dtype=np.float32),
        "b": np.array([0.0, 1.0, 0.0], dtype=np.float32),
    }
    overlap = batch_style_overlap(["a", "b"], style_map, torch.device("cpu"))
    assert overlap is not None
    assert overlap.shape == (2, 2)
    assert float(overlap[0, 0]) == pytest.approx(1.0)
    assert float(overlap[1, 1]) == pytest.approx(1.0)
    assert float(overlap[0, 1]) == pytest.approx(0.0)


def test_command_overlap_infonce_prefers_matching_style_targets():
    z = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
    targets = torch.tensor([[0.9, 0.1], [0.1, 0.9]])
    aligned = command_overlap_infonce(z, z, targets, temperature=0.1, target_temperature=0.2)
    shuffled = command_overlap_infonce(z, z, targets.flip(1), temperature=0.1, target_temperature=0.2)
    assert float(aligned) < float(shuffled)


def test_example_pool_retrieve_respects_kind_quota_and_backfill():
    examples = [
        [PoolExample("sim one", "sim_contrast"), PoolExample("human one", "human_contrast")],
        [PoolExample("sim two", "sim_contrast"), PoolExample("human two", "human_contrast")],
    ]
    vectors = np.array([[1.0, 0.0], [0.9, 0.1]], dtype=np.float32)
    index = EmbeddingIndex(2)
    index.add(vectors)
    pool = ExamplePool(["s1", "s2"], vectors, examples, index)
    out = pool.retrieve([1.0, 0.0], k_sessions=2, n_general=1, n_specific=1)
    assert len(out) == 2
    assert "sim one" in out or "sim two" in out
    assert "human one" in out or "human two" in out


def test_examples_from_manual_filters_meta_examples():
    manual = validate_manual(
        {
            "commands": [
                {
                    "text": "Use short replies.",
                    "kind": "human_contrast",
                    "form": "general",
                    "examples": ['assistant asks "where?" user says "nyc"'],
                },
                {
                    "text": "Stay terse.",
                    "kind": "sim_contrast",
                    "form": "general",
                    "examples": ["ok"],
                },
            ]
        }
    )
    exs = examples_from_manual(manual)
    texts = {e.text for e in exs}
    assert "ok" in texts
    assert all("assistant asks" not in t for t in texts)


def test_eval_validation_retrieval_reports_gap():
    gallery_examples = [
        [PoolExample("book flight", "human_contrast")],
        [PoolExample("cancel it", "human_contrast")],
        [PoolExample("change seat", "human_contrast")],
        [PoolExample("add bag", "human_contrast")],
    ]
    gallery_vectors = np.eye(4, dtype=np.float32)
    query_vectors = gallery_vectors[3:]
    metrics = eval_validation_retrieval(
        gallery_ids=["g0", "g1", "g2"],
        gallery_vectors=gallery_vectors[:3],
        gallery_examples=gallery_examples[:3],
        query_ids=["q3"],
        query_vectors=query_vectors,
        query_target_examples=[["add bag"]],
        k_sessions=2,
        n_general=0,
        n_specific=1,
    )
    assert metrics["retrieval_overlap"] >= metrics["retrieval_shuffled_overlap"]
    assert "retrieval_gap" in metrics


def test_example_pool_roundtrip(tmp_path: Path):
    examples = [[PoolExample("hello", "human_contrast")]]
    vectors = np.array([[1.0, 0.0, 0.0]], dtype=np.float32)
    index = EmbeddingIndex(3)
    index.add(vectors)
    pool = ExamplePool(["s1"], vectors, examples, index, "ckpt")
    out_dir = tmp_path / "pool"
    pool.save(out_dir)
    loaded = ExamplePool.load(out_dir)
    assert loaded.session_ids == ["s1"]
    assert loaded.examples[0][0].text == "hello"
    assert (out_dir / "manifest.json").is_file()


def test_inject_examples_forwards_k_sessions(monkeypatch):
    from cue_training.data import example_pool as ep

    captured: dict[str, Any] = {}

    def _fake_augment(general, specific, **kwargs):
        captured.update(kwargs)
        return general, {**specific, "examples": ["span"]}, ["span"]

    monkeypatch.setattr(ep, "augment_manual_block", _fake_augment)
    examples = [[PoolExample("hello", "human_contrast")]]
    vectors = np.array([[1.0, 0.0]], dtype=np.float32)
    index = EmbeddingIndex(2)
    index.add(vectors)
    pool = ExamplePool(["s1"], vectors, examples, index)
    rendered = ep.inject_examples_into_dual_render(
        '{"commands":["Be terse."],"examples":[]}',
        '{"commands":["Ask once."],"examples":[]}',
        [1.0, 0.0],
        pool,
        k_sessions=5,
        n_general=1,
        n_specific=1,
    )
    assert captured["k_sessions"] == 5
    assert "Ask once." in rendered
    assert "span" in rendered


def test_example_retrieval_cli_and_sidecar_args():
    from cue_training.evaluation.common.user_sims import example_retrieval_cli_flags, example_retrieval_kwargs
    from cue_training.evaluation.rollouts.sidecar import sidecar_args_for_job

    assert example_retrieval_cli_flags(enabled=False) == [
        "--example_retrieval_k_sessions=8",
        "--example_retrieval_n_general=2",
        "--example_retrieval_n_specific=2",
    ]
    assert "--example_retrieval_enabled" in example_retrieval_cli_flags(enabled=True)
    assert example_retrieval_kwargs(
        {"example_retrieval_enabled": True, "example_retrieval_k_sessions": 12}
    ) == {
        "example_retrieval_enabled": True,
        "example_retrieval_k_sessions": 12,
        "example_retrieval_n_general": 2,
        "example_retrieval_n_specific": 2,
    }
    args = sidecar_args_for_job(
        variant="decoder",
        sim_model="gpt-5.4-mini",
        sim_api_base=None,
        sim_api_key_env="OPENAI_API_KEY",
        decoder_dir="/tmp/ckpt",
        example_retrieval={
            "enabled": True,
            "k_sessions": 8,
            "n_general": 2,
            "n_specific": 2,
        },
    )
    assert "--example_retrieval_enabled" in args
    assert "--example_retrieval_k_sessions=8" in args
