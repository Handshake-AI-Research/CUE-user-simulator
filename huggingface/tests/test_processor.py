"""CueProcessor chat normalization and session preprocessing."""

from __future__ import annotations

import pytest
from cue_hf import CueConfig, CueProcessor
from cue_hf.session_preprocess import DRAFT_STUB


def test_single_session_becomes_batch_of_one():
    out = CueProcessor()([{"role": "user", "content": "hi"}])
    assert out == [[{"role": "user", "content": "hi"}]]


def test_batch_of_sessions_is_preserved():
    out = CueProcessor()(
        [
            [{"role": "user", "content": "a"}],
            [{"role": "user", "content": "b"}, {"role": "assistant", "content": "c"}],
        ]
    )
    assert [len(session) for session in out] == [1, 2]


def test_bare_string_is_one_user_turn():
    assert CueProcessor()("just this") == [[{"role": "user", "content": "just this"}]]


def test_content_parts_are_joined_and_unknown_roles_dropped():
    out = CueProcessor()(
        [
            {"role": "tool", "content": "ignored"},
            {"role": "user", "content": [{"type": "text", "text": "part one"}, {"type": "text", "text": "part two"}]},
        ]
    )
    assert out == [[{"role": "user", "content": "part one part two"}]]


def test_empty_session_is_rejected():
    with pytest.raises(ValueError, match="no usable"):
        CueProcessor()([{"role": "tool", "content": "nope"}])


def test_user_only_drops_assistant_turns():
    out = CueProcessor(session_preprocess="user_only")(
        [
            {"role": "user", "content": "hi"},
            {"role": "assistant", "content": "hello there"},
        ]
    )
    assert out == [[{"role": "user", "content": "hi"}]]


def test_strip_document_replaces_long_draft_with_stub():
    draft = "Subject: Quarterly update\n\n" + " ".join(["body"] * 200)
    out = CueProcessor()(
        [{"role": "user", "content": "write it"}, {"role": "assistant", "content": draft}],
        session_preprocess="strip_document",
    )
    assert out[0][1]["content"] == DRAFT_STUB
    # The same turn survives untouched in the default mode.
    kept = CueProcessor()([{"role": "user", "content": "write it"}, {"role": "assistant", "content": draft}])
    assert kept[0][1]["content"] == draft


def test_from_pretrained_adopts_the_config_default(tmp_path):
    CueConfig(session_preprocess="strip_document").save_pretrained(tmp_path)
    assert CueProcessor.from_pretrained(tmp_path).session_preprocess == "strip_document"
    override = CueProcessor.from_pretrained(tmp_path, session_preprocess="user_only")
    assert override.session_preprocess == "user_only"
    # Private repos need the hub kwargs to reach the config load, not the constructor.
    assert CueProcessor.from_pretrained(tmp_path, token=False).session_preprocess


def test_invalid_mode_is_rejected():
    with pytest.raises(ValueError, match="session_preprocess"):
        CueProcessor(session_preprocess="sometimes")
