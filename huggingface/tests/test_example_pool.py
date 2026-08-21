"""Example-pool retrieval quotas, injection, and on-disk loading."""

from __future__ import annotations

import json

import numpy as np
from cue_hf.example_pool import ExamplePool, PoolExample, inject_examples


def _pool() -> ExamplePool:
    vectors = np.eye(3, dtype=np.float32)
    return ExamplePool.from_rows(
        ["s0", "s1", "s2"],
        vectors,
        [
            [
                PoolExample("general zero", "sim_contrast"),
                PoolExample("specific zero", "human_contrast"),
            ],
            [PoolExample("style one", "style")],
            [PoolExample("specific two", "human_contrast")],
        ],
    )


def test_retrieve_respects_kind_quotas():
    out = _pool().retrieve([1.0, 0.0, 0.0], k_sessions=3, n_general=1, n_specific=1)
    assert out == ["general zero", "specific zero"]


def test_retrieve_orders_by_cosine_similarity():
    out = _pool().retrieve([0.0, 1.0, 0.0], k_sessions=3, n_general=0, n_specific=1)
    assert out == ["style one"]


def test_retrieve_can_exclude_the_query_session():
    out = _pool().retrieve([1.0, 0.0, 0.0], k_sessions=3, n_general=0, n_specific=2, exclude_session_id="s0")
    assert out == ["style one", "specific two"]


def test_exclusion_only_applies_to_the_quota_pass():
    """Upstream's underfill backfill rescans every neighbor, excluded ones included."""

    out = _pool().retrieve([1.0, 0.0, 0.0], k_sessions=3, n_general=1, n_specific=1, exclude_session_id="s0")
    assert "general zero" in out


def test_examples_dropped_by_a_full_quota_are_not_backfilled():
    """Upstream marks an example seen before the quota check, so it can underfill."""

    out = _pool().retrieve([1.0, 0.0, 0.0], k_sessions=3, n_general=3, n_specific=1)
    assert out == ["general zero", "specific zero"]


def test_inject_examples_replaces_manual_examples():
    manual = {"commands": ["be terse"], "examples": ["decoded example"]}
    out = inject_examples(manual, pool=_pool(), cue_embedding=[1.0, 0.0, 0.0], n_general=1, n_specific=1)
    assert out["commands"] == ["be terse"]
    assert out["examples"] == ["general zero", "specific zero"]


def test_inject_examples_passes_through_none():
    assert inject_examples(None, pool=_pool(), cue_embedding=[1.0, 0.0, 0.0]) is None


def test_from_directory_reads_cue_training_layout(tmp_path):
    pool = _pool()
    directory = tmp_path / "example_pool"
    directory.mkdir()
    np.save(directory / "cue_vectors.npy", pool.cue_vectors)
    pool.index.save(directory / "index.faiss")
    with (directory / "meta.jsonl").open("w", encoding="utf-8") as handle:
        for session_id, examples in zip(pool.session_ids, pool.examples, strict=True):
            handle.write(
                json.dumps(
                    {
                        "session_id": session_id,
                        "examples": [{"text": e.text, "kind": e.kind} for e in examples],
                    }
                )
                + "\n"
            )
    (directory / "manifest.json").write_text(json.dumps({"checkpoint_fingerprint": "abc123"}), encoding="utf-8")

    # Both the pool dir and its parent checkpoint dir resolve.
    for source in (directory, tmp_path):
        loaded = ExamplePool.from_directory(source)
        assert loaded.session_ids == pool.session_ids
        assert loaded.checkpoint_fingerprint == "abc123"
        assert loaded.retrieve([1.0, 0.0, 0.0], n_general=1, n_specific=1) == [
            "general zero",
            "specific zero",
        ]
