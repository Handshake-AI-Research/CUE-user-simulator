import pytest

from cue_training.preprocessing.delexicalize import delexicalize_text
from cue_training.preprocessing.gates import (
    ManualGateError,
    detect_answer_lag,
    detect_session_signatures,
    gate_manual,
    ground_evidence_turn_ids,
)


def _cmd(text: str, kind: str = "sim_contrast", evidence=None, examples=None):
    return {
        "text": text,
        "kind": kind,
        "form": "general",
        "examples": examples or ["short reply"],
        "evidence_turn_ids": evidence or [],
    }


def _laggy_turns():
    """Assistant asks name → membership → date; user answers previous asks (lag)."""
    return [
        {"role": "assistant", "turn_id": "a0", "content": "Can I get your full name?"},
        {"role": "user", "turn_id": "u0", "content": "I need a return"},
        {"role": "assistant", "turn_id": "a1", "content": "What is your membership level?"},
        {"role": "user", "turn_id": "u1", "content": "Chloe Zhang"},
        {"role": "assistant", "turn_id": "a2", "content": "What is the purchase date?"},
        {"role": "user", "turn_id": "u2", "content": "Bronze"},
        {"role": "assistant", "turn_id": "a3", "content": "Anything else today?"},
        {"role": "user", "turn_id": "u3", "content": "12/18/2019"},
    ]


def _ok_cmds(evidence="u0", lag_cmd=False):
    cmds = [_cmd(f"Use short casual replies {i}", evidence=[evidence]) for i in range(5)]
    human = [
        _cmd("If asked for a detail late, keep the reply compact.", kind="human_contrast", evidence=[evidence]),
        _cmd("If closing, thank briefly.", kind="human_contrast", evidence=[evidence]),
        _cmd("If confused, ask one clarifying question.", kind="human_contrast", evidence=[evidence]),
        _cmd("Do not invent side topics mid-flow.", kind="human_contrast", evidence=[evidence]),
    ]
    if lag_cmd:
        human[0] = _cmd(
            "If the assistant asks a new question, answer the previous question first (answer lag).",
            kind="human_contrast",
            evidence=["u1", "u2", "u3"],
            examples=["reply to the previous prompt"],
        )
    else:
        human.append(
            _cmd(
                "Maintain a steady, cooperative back-and-forth that follows the assistant's sequence.",
                kind="human_contrast",
                evidence=[evidence],
            )
        )
        return cmds + human
    human.append(_cmd("Close with a short acknowledgment.", kind="human_contrast", evidence=[evidence]))
    return cmds + human


def test_ground_evidence_fills_from_overlap():
    turns = [
        {"role": "user", "turn_id": "u0", "content": "I keep interrupting with side questions."},
        {"role": "assistant", "turn_id": "a0", "content": "Sure."},
        {"role": "user", "turn_id": "u1", "content": "Also I answer with one word."},
    ]
    manual = {
        "schema_version": 1,
        "commands": [_cmd("Interrupt with side questions before answering.", evidence=[])],
    }
    out = ground_evidence_turn_ids(manual, turns)
    assert out["commands"][0]["evidence_turn_ids"]
    assert "u0" in out["commands"][0]["evidence_turn_ids"]


def test_gate_rejects_content_leak():
    turns = [{"role": "user", "turn_id": "u0", "content": "hi"}]
    cmds = [_cmd(f"rule {i}", evidence=["u0"]) for i in range(5)]
    cmds += [
        _cmd("Mention the Guess shirt when asked.", kind="human_contrast", evidence=["u0"])
        for _ in range(5)
    ]
    with pytest.raises(ManualGateError, match="content leak"):
        gate_manual({"schema_version": 1, "commands": cmds}, turns=turns)


def test_gate_rejects_generic_human_contrast():
    turns = [{"role": "user", "turn_id": "u0", "content": "okay thanks"}]
    cmds = [_cmd(f"Avoid assistant hedging {i}", evidence=["u0"]) for i in range(5)]
    cmds += [
        _cmd("Be polite and cooperative.", kind="human_contrast", evidence=["u0"]),
        _cmd("Stay calm and matter-of-fact.", kind="human_contrast", evidence=["u0"]),
        _cmd("Be task-focused without adding extras.", kind="human_contrast", evidence=["u0"]),
        _cmd("Acknowledge it briefly.", kind="human_contrast", evidence=["u0"]),
        _cmd("Use brief courtesy markers.", kind="human_contrast", evidence=["u0"]),
    ]
    with pytest.raises(ManualGateError, match="too generic"):
        gate_manual({"schema_version": 1, "commands": cmds}, turns=turns)


def test_gate_rejects_frustration_vs_acceptance():
    turns = [{"role": "user", "turn_id": "u0", "content": "okay thanks no problem"}]
    cmds = [_cmd(f"Use short replies {i}", evidence=["u0"]) for i in range(5)]
    cmds += [
        _cmd(
            "Show frustration when the agent repeats itself.",
            kind="human_contrast",
            evidence=["u0"],
        )
        for _ in range(5)
    ]
    with pytest.raises(ManualGateError, match="frustration"):
        gate_manual({"schema_version": 1, "commands": cmds}, turns=turns)


def test_delexicalize_strips_task_nouns():
    assert "<TERM>" in delexicalize_text("Ask about the Guess shirt and membership gold")
    assert "<TERM>" in delexicalize_text("reset my password")


def test_detect_answer_lag_finds_out_of_order_replies():
    events = detect_answer_lag(_laggy_turns())
    assert len(events) >= 2
    lagged_ids = {e["user_turn_id"] for e in events}
    assert "u1" in lagged_ids
    assert "u2" in lagged_ids or "u3" in lagged_ids


def test_gate_rejects_missing_lag_behavior():
    turns = _laggy_turns()
    # Sequential claim + no lag capture should fail.
    with pytest.raises(ManualGateError, match="answer lag"):
        gate_manual({"schema_version": 1, "commands": _ok_cmds(lag_cmd=False)}, turns=turns)


def test_gate_rejects_unmentioned_repeated_lag():
    turns = _laggy_turns()
    cmds = [_cmd(f"Use short casual replies {i}", evidence=["u0"]) for i in range(5)]
    cmds += [
        _cmd("If asked for details, keep replies compact.", kind="human_contrast", evidence=["u0"]),
        _cmd("If closing, thank briefly.", kind="human_contrast", evidence=["u0"]),
        _cmd("If confused, ask one clarifying question.", kind="human_contrast", evidence=["u0"]),
        _cmd("Do not invent side topics mid-flow.", kind="human_contrast", evidence=["u0"]),
        _cmd("Close with a short acknowledgment.", kind="human_contrast", evidence=["u0"]),
    ]
    with pytest.raises(ManualGateError, match="out-of-order"):
        gate_manual({"schema_version": 1, "commands": cmds}, turns=turns)


def test_gate_accepts_manual_with_lag_command():
    turns = _laggy_turns()
    gate_manual({"schema_version": 1, "commands": _ok_cmds(lag_cmd=True)}, turns=turns)


def test_gate_accepts_lag_synonyms():
    turns = _laggy_turns()
    for phrasing in [
        "Answer the assistant's earlier question even when the conversation has already moved on.",
        "Reply in a delayed, out-of-order way to a later requested detail.",
        "Follow the conversation's lagged order rather than the latest prompt.",
        "Keep answering the older thread instead of the newest question.",
    ]:
        cmds = [_cmd(f"Use short casual replies {i}", evidence=["u0"]) for i in range(5)]
        cmds += [
            _cmd(phrasing, kind="human_contrast", evidence=["u1", "u2"]),
            _cmd("If closing, thank briefly.", kind="human_contrast", evidence=["u0"]),
            _cmd("If confused, ask one clarifying question.", kind="human_contrast", evidence=["u0"]),
            _cmd("Do not invent side topics mid-flow.", kind="human_contrast", evidence=["u0"]),
            _cmd("Close with a short acknowledgment.", kind="human_contrast", evidence=["u0"]),
        ]
        gate_manual({"schema_version": 1, "commands": cmds}, turns=turns)


def test_gate_rejects_sequential_synonyms_when_lag_present():
    turns = _laggy_turns()
    cmds = [_cmd(f"Use short casual replies {i}", evidence=["u0"]) for i in range(5)]
    cmds += [
        _cmd(
            "If the assistant asks a new question, answer the previous question first (answer lag).",
            kind="human_contrast",
            evidence=["u1", "u2"],
        ),
        _cmd(
            "When the assistant asks a follow-up, do not answer it out of order.",
            kind="human_contrast",
            evidence=["u0"],
        ),
        _cmd("If closing, thank briefly.", kind="human_contrast", evidence=["u0"]),
        _cmd("If confused, ask one clarifying question.", kind="human_contrast", evidence=["u0"]),
        _cmd("Close with a short acknowledgment.", kind="human_contrast", evidence=["u0"]),
    ]
    with pytest.raises(ManualGateError, match="sequential"):
        gate_manual({"schema_version": 1, "commands": cmds}, turns=turns)


def _escalation_turns():
    return [
        {"role": "assistant", "content": "I cannot extend that policy."},
        {"role": "user", "turn_id": "u0", "content": "Could you speak with a manager?"},
        {"role": "assistant", "content": "I will have a manager call you. Phone?"},
        {"role": "user", "turn_id": "u1", "content": "206-555-0100"},
    ]


def _polite_close_turns():
    return [
        {"role": "assistant", "content": "Anything else today?"},
        {"role": "user", "turn_id": "n0", "content": "No thanks that is all"},
        {"role": "assistant", "content": "Have a great day"},
        {"role": "user", "turn_id": "n1", "content": "Thank you"},
    ]


def test_detect_session_signatures_finds_escalation():
    assert "escalation" in detect_session_signatures(_escalation_turns())
    assert "answer_lag" in detect_session_signatures(_laggy_turns())


def test_gate_requires_distinctive_human_vs_negatives():
    target = _laggy_turns()
    # Add escalation so target has 2 available signatures vs polite negatives.
    target = target + [
        {"role": "assistant", "content": "I cannot help further."},
        {"role": "user", "turn_id": "u4", "content": "Please get a manager"},
    ]
    negatives = [_polite_close_turns() for _ in range(5)]
    # Mentions lag (passes lag gate) but only 1 TARGET-only signature claimed.
    soft = [_cmd(f"Use short casual replies {i}", evidence=["u0"]) for i in range(5)]
    soft += [
        _cmd(
            "If the assistant asks a new question, answer the previous question first (answer lag).",
            kind="human_contrast",
            evidence=["u1", "u2"],
        ),
        _cmd("If closing, thank briefly.", kind="human_contrast", evidence=["u0"]),
        _cmd("If asked whether anything else, say no.", kind="human_contrast", evidence=["u0"]),
        _cmd("If confused, ask one clarifying question.", kind="human_contrast", evidence=["u0"]),
        _cmd("Close with a short acknowledgment.", kind="human_contrast", evidence=["u0"]),
    ]
    with pytest.raises(ManualGateError, match="TARGET-only"):
        gate_manual({"schema_version": 1, "commands": soft}, turns=target, negatives=negatives)

    good = [_cmd(f"Use short casual replies {i}", evidence=["u0"]) for i in range(5)]
    good += [
        _cmd(
            "If the assistant asks a new question, answer the previous question first (answer lag).",
            kind="human_contrast",
            evidence=["u1", "u2"],
        ),
        _cmd(
            "If refused, escalate by asking for a manager.",
            kind="human_contrast",
            evidence=["u4"],
        ),
        _cmd("If closing, thank briefly.", kind="human_contrast", evidence=["u0"]),
        _cmd("If confused, ask one clarifying question.", kind="human_contrast", evidence=["u0"]),
        _cmd("Close with a short acknowledgment.", kind="human_contrast", evidence=["u0"]),
    ]
    gate_manual({"schema_version": 1, "commands": good}, turns=target, negatives=negatives)


def test_gate_skips_distinctiveness_when_target_has_no_unique_sigs():
    # Thin polite target vs same-style negatives → available empty → no distinctiveness fail.
    turns = _polite_close_turns()
    negatives = [_polite_close_turns() for _ in range(5)]
    cmds = [_cmd(f"Use short casual replies {i}", evidence=["n0"]) for i in range(5)]
    cmds += [
        _cmd("If closing, thank briefly.", kind="human_contrast", evidence=["n0"]),
        _cmd("If asked whether anything else, say no.", kind="human_contrast", evidence=["n0"]),
        _cmd("If confused, ask one clarifying question.", kind="human_contrast", evidence=["n0"]),
        _cmd("Do not invent side topics mid-flow.", kind="human_contrast", evidence=["n0"]),
        _cmd("Close with a short acknowledgment.", kind="human_contrast", evidence=["n1"]),
    ]
    gate_manual({"schema_version": 1, "commands": cmds}, turns=turns, negatives=negatives)
