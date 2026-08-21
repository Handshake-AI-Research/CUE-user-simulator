"""Unit tests for tau2 rich conversation + reward_report export (no tau2 runtime)."""

from __future__ import annotations

from types import SimpleNamespace

from cue_training.evaluation.common.mirrorbench_export import strip_tool_turns, to_mirrorbench_rollout
from cue_training.evaluation.tau2_bench.bridge import _message_dicts, reward_report_from_info


def test_message_dicts_serializes_tool_calls_and_results() -> None:
    sim = SimpleNamespace(
        messages=[
            SimpleNamespace(role="user", content="Hi", tool_calls=None, raw_data=None),
            SimpleNamespace(
                role="assistant",
                content="",
                tool_calls=[
                    SimpleNamespace(
                        id="call_1",
                        name="get_user_details",
                        arguments={"user_id": "u1"},
                        requestor="assistant",
                    )
                ],
                raw_data=None,
            ),
            SimpleNamespace(
                role="tool",
                id="call_1",
                content='{"name": "Ada"}',
                requestor="assistant",
                error=False,
                tool_calls=None,
                raw_data=None,
            ),
            SimpleNamespace(
                role="assistant",
                content="Found you.",
                tool_calls=None,
                raw_data=None,
            ),
        ]
    )
    turns = _message_dicts(sim)
    assert turns[1]["tool_calls"] == [
        {
            "id": "call_1",
            "name": "get_user_details",
            "arguments": {"user_id": "u1"},
            "requestor": "assistant",
        }
    ]
    assert turns[2]["role"] == "tool"
    assert turns[2]["tool_call_id"] == "call_1"
    assert turns[2]["content"] == '{"name": "Ada"}'
    # Metrics path still drops tools / empty assistant tool-only turns.
    cleaned = strip_tool_turns(turns)
    assert [t["role"] for t in cleaned] == ["user", "assistant"]
    assert cleaned[1]["content"] == "Found you."


def test_reward_report_failed_checks_only() -> None:
    info = SimpleNamespace(
        reward=0.0,
        reward_breakdown={"DB": 0.0, "COMMUNICATE": 1.0},
        reward_basis=["DB", "COMMUNICATE"],
        db_check=SimpleNamespace(db_match=False, db_reward=0.0),
        action_checks=[
            SimpleNamespace(
                action_match=True,
                action_reward=1.0,
                tool_type="read",
                action=SimpleNamespace(name="ok", arguments={}, action_id="a0"),
            ),
            SimpleNamespace(
                action_match=False,
                action_reward=0.0,
                tool_type="write",
                action=SimpleNamespace(
                    name="cancel_reservation",
                    arguments={"reservation_id": "r1"},
                    action_id="a1",
                ),
            ),
        ],
        nl_assertions=[
            SimpleNamespace(met=True, nl_assertion="said bye", justification="ok"),
            SimpleNamespace(met=False, nl_assertion="confirmed refund", justification="never said"),
        ],
        communicate_checks=[
            SimpleNamespace(met=False, info="confirmation number", justification="missing"),
        ],
        env_assertions=[
            SimpleNamespace(met=True, reward=1.0, env_assertion={"fn": "ok"}),
            SimpleNamespace(met=False, reward=0.0, env_assertion={"fn": "assert_status"}),
        ],
    )
    report = reward_report_from_info(info)
    assert report is not None
    assert report["reward"] == 0.0
    kinds = [f["kind"] for f in report["failed"]]
    assert kinds == ["db", "action", "nl_assertion", "communicate", "env_assertion"]
    action = next(f for f in report["failed"] if f["kind"] == "action")
    assert action["name"] == "cancel_reservation"
    assert action["arguments"] == {"reservation_id": "r1"}


def test_mirrorbench_keeps_rich_fields_in_metadata() -> None:
    conversation = [
        {"role": "user", "content": "help"},
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [{"id": "c1", "name": "lookup", "arguments": {}, "requestor": "assistant"}],
        },
        {"role": "tool", "tool_call_id": "c1", "content": "ok"},
        {"role": "assistant", "content": "done"},
    ]
    report = {"reward": 0.0, "failed": [{"kind": "db", "passed": False}]}
    record = to_mirrorbench_rollout(
        source_record={"episode_id": "e1", "task_id": "t1", "dataset": "airline"},
        conversation=conversation,
        provenance={
            "reward": 0.0,
            "is_success": False,
            "reward_report": report,
            "full_conversation": conversation,
        },
    )
    meta = record["metadata"]
    assert meta["reward_report"] == report
    assert meta["full_conversation"][1]["tool_calls"][0]["name"] == "lookup"
    cleaned = record["rollout_conversation"]
    assert [t["role"] for t in cleaned] == ["user", "assistant"]
    assert cleaned[1]["content"] == "done"
    assert all("tool_calls" not in t for t in cleaned)
