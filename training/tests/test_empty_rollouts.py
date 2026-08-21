"""Empty-user rollout detection for failed sim calls."""

from __future__ import annotations

from cue_training.evaluation.common.mirrorbench_export import (
    has_nonempty_user_turn,
    is_empty_rollout_record,
)


def test_has_nonempty_user_turn():
    assert not has_nonempty_user_turn([])
    assert not has_nonempty_user_turn([{"role": "user", "content": ""}])
    assert not has_nonempty_user_turn([{"role": "assistant", "content": "hi"}])
    assert has_nonempty_user_turn([{"role": "user", "content": "help"}])


def test_is_empty_rollout_record():
    assert is_empty_rollout_record({"rollout_conversation": []})
    assert is_empty_rollout_record(
        {"rollout_conversation": [{"role": "assistant", "content": "ok"}]}
    )
    assert not is_empty_rollout_record(
        {"rollout_conversation": [{"role": "user", "content": "hi"}]}
    )
