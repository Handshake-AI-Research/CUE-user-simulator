"""Convert a simulated conversation into a MirrorBench-loadable rollout record.

MirrorBench's ``dataset:jsonl/rollout`` loader expects plain user/assistant text
turns, so any tool calls / tool (environment) messages are stripped here -- otherwise
tool-call payloads confuse the pipeline. The cleaned transcript is wrapped into a
rollout record via ``evaluation.common.rollout.rollout_record_from_conversation`` and
tagged with provenance so downstream MirrorBench metrics can be joined back to the
native harness result (e.g. tau2 reward).

Rich tool-aware transcripts and ``reward_report`` (when present) live under
``metadata`` for Failure Mode Analysis; they are not used by MirrorBench metrics.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

Turn = dict[str, Any]

_USER_ROLES = {"user", "human", "customer"}
_ASSISTANT_ROLES = {"assistant", "agent", "bot", "chatbot", "model"}


def strip_tool_turns(conversation: list[Turn]) -> list[dict[str, Any]]:
    """Keep only user + assistant natural-language turns (drop tools/tool-calls)."""

    cleaned: list[dict[str, Any]] = []
    for turn in conversation or []:
        if not isinstance(turn, Mapping):
            continue
        role = str(turn.get("role") or "").lower()
        if role in _USER_ROLES:
            role = "user"
        elif role in _ASSISTANT_ROLES:
            role = "assistant"
        else:
            # tool / environment / system messages are dropped
            continue
        content = str(turn.get("content") or "").strip()
        if not content:
            # e.g. an assistant message that only carried tool_calls
            continue
        entry: dict[str, Any] = {"role": role, "content": content}
        if turn.get("raw_data"):  # per-turn provenance (e.g. hard-decoder command block)
            entry["raw_data"] = turn["raw_data"]
        cleaned.append(entry)
    return cleaned


def has_nonempty_user_turn(conversation: list[Turn] | None) -> bool:
    """True if the dialogue has at least one non-empty user utterance.

    Used to reject failed user-sim calls that return ``("", done=True)`` and would
    otherwise be written as valid empty rollouts.
    """

    for turn in strip_tool_turns(list(conversation or [])):
        if turn.get("role") == "user":
            return True
    return False


def is_empty_rollout_record(row: Mapping[str, Any]) -> bool:
    """True if a MirrorBench rollout JSONL row has no usable user turns."""

    conv = row.get("rollout_conversation")
    if not isinstance(conv, list):
        return True
    return not has_nonempty_user_turn(conv)


def to_mirrorbench_rollout(
    *,
    source_record: Mapping[str, Any],
    conversation: list[Turn],
    provenance: Mapping[str, Any],
) -> dict[str, Any]:
    """Build a MirrorBench rollout record from a (possibly tool-laden) conversation."""

    from cue_training.evaluation.common.rollout import rollout_record_from_conversation

    clean = strip_tool_turns(conversation)
    record = rollout_record_from_conversation(source_record, clean)
    metadata = record.setdefault("metadata", {})
    metadata.update({k: v for k, v in provenance.items() if v is not None})
    return record
