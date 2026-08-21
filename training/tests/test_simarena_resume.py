"""Tests for SimulatorArena episode-level resume + streamed JSONL writes."""

from __future__ import annotations

import json
from pathlib import Path

from cue_training.evaluation.common.io import append_jsonl, read_jsonl
from cue_training.evaluation.simulatorarena.runner import (
    _episode_key,
    _is_terminal_result,
    _rebuild_native_from_rollouts,
    prepare_resume,
)


def test_terminal_result_keeps_ok_and_content_policy() -> None:
    assert _is_terminal_result({"status": "ok"})
    assert _is_terminal_result({"status": "error", "content_policy_blocked": True})
    assert not _is_terminal_result({"status": "error", "error": "empty_user_sim_response"})
    assert not _is_terminal_result({"status": "error", "content_policy_blocked": False})


def test_prepare_resume_compacts_and_skips_terminal(tmp_path: Path) -> None:
    results = tmp_path / "results.jsonl"
    rollouts = tmp_path / "rollout.simulatorarena.jsonl"
    rows = [
        {"variant": "baseline:realusersim", "arm": "as_is", "episode_id": "a", "status": "ok"},
        {
            "variant": "baseline:realusersim",
            "arm": "as_is",
            "episode_id": "b",
            "status": "error",
            "error": "empty_user_sim_response",
        },
        {
            "variant": "baseline:realusersim",
            "arm": "as_is",
            "episode_id": "c",
            "status": "error",
            "content_policy_blocked": True,
        },
    ]
    results.write_text("\n".join(json.dumps(r) for r in rows) + "\n", encoding="utf-8")
    rollouts.write_text(
        "\n".join(
            [
                json.dumps({
                    "episode_id": "a",
                    "rollout_conversation": [{"role": "user", "content": "hi"}],
                    "metadata": {
                        "variant": "baseline:realusersim",
                        "arm": "as_is",
                        "assistant_model": "m",
                        "workerId": "w1",
                        "document_type": "email",
                        "intent": "x",
                    },
                }),
                json.dumps({
                    "episode_id": "orphan",
                    "rollout_conversation": [{"role": "user", "content": "nope"}],
                    "metadata": {
                        "variant": "baseline:realusersim",
                        "arm": "as_is",
                    },
                }),
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    done = prepare_resume(tmp_path, keep_rollouts=True)
    assert done == {
        _episode_key("baseline:realusersim", "as_is", "a"),
        _episode_key("baseline:realusersim", "as_is", "c"),
    }
    kept_results = read_jsonl(str(results))
    assert [r["episode_id"] for r in kept_results] == ["a", "c"]
    kept_rollouts = read_jsonl(str(rollouts))
    assert [r["episode_id"] for r in kept_rollouts] == ["a"]


def test_append_jsonl_streams_without_rewrite(tmp_path: Path) -> None:
    path = tmp_path / "out.jsonl"
    append_jsonl(str(path), [{"episode_id": "1"}])
    append_jsonl(str(path), [{"episode_id": "2"}])
    assert [r["episode_id"] for r in read_jsonl(str(path))] == ["1", "2"]


def test_rebuild_native_from_rollouts(tmp_path: Path) -> None:
    rollouts = tmp_path / "rollout.simulatorarena.jsonl"
    rollouts.write_text(
        json.dumps({
            "episode_id": "ep1",
            "rollout_conversation": [
                {"role": "user", "content": "hello"},
                {"role": "assistant", "content": "hi"},
            ],
            "metadata": {
                "variant": "baseline:realusersim",
                "arm": "as_is",
                "assistant_model": "gpt-x",
                "workerId": "worker-1",
                "document_type": "email",
                "intent": "reply",
            },
        })
        + "\n",
        encoding="utf-8",
    )
    _rebuild_native_from_rollouts(tmp_path, "writing")
    native = json.loads(
        (tmp_path / "native" / "baseline:realusersim__as_is.json").read_text(encoding="utf-8")
    )
    assert "gpt-x" in native
    assert "Email/Letter" in native["gpt-x"] or "email" in native["gpt-x"]
