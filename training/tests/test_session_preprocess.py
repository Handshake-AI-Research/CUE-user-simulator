"""Tests for encode-time session preprocess (strip_document / user_only)."""

from __future__ import annotations

import json

import pytest
import torch

from cue_training.infer import export
from cue_training.infer.session_preprocess import (
    DRAFT_STUB,
    preprocess_turns,
    strip_assistant_document,
)


def test_strip_keeps_preamble_drops_draft_body():
    text = (
        "Sure — here's a revised draft with a warmer tone.\n\n"
        "Subject: Team lunch\n\n"
        "Hi all,\n\n"
        "Looking forward to seeing everyone next week at the park for our annual picnic "
        "celebration with food, games, and plenty of time to catch up with colleagues."
    )
    out = strip_assistant_document(text)
    assert "Sure" in out
    assert "Subject:" not in out
    assert "Looking forward" not in out


def test_strip_keeps_lead_in_before_dear_marker():
    text = (
        "Got it! Here's the revised invite:\n"
        "Dear Jordan,\n\n"
        "Please join us for dinner on Friday evening at seven."
    )
    out = strip_assistant_document(text)
    assert "Got it" in out
    assert "Dear Jordan" not in out


def test_strip_stubs_draft_only():
    text = (
        "Subject: Quarterly update\n\n"
        "Dear team,\n\n"
        "This quarter we shipped several features and improved reliability across "
        "the platform while continuing to invest in documentation and onboarding."
    )
    assert strip_assistant_document(text) == DRAFT_STUB


def test_strip_keeps_short_clarify():
    text = "Do you want this more formal, or should I keep the casual tone?"
    assert strip_assistant_document(text) == text


def test_preprocess_turns_modes():
    turns = [
        {"role": "user", "content": "make it shorter"},
        {
            "role": "assistant",
            "content": (
                "Sure, here's a shorter version.\n\n"
                "Dear Sam,\n\nThanks for your note about the project timeline."
            ),
        },
        {"role": "user", "content": "even shorter"},
    ]
    full = preprocess_turns(turns, "full")
    assert full[1]["content"].startswith("Sure")
    assert "Dear Sam" in full[1]["content"]

    stripped = preprocess_turns(turns, "strip_document")
    assert "Sure" in stripped[1]["content"]
    assert "Dear Sam" not in stripped[1]["content"]
    assert stripped[0]["content"] == "make it shorter"

    user_only = preprocess_turns(turns, "user_only")
    assert [t["role"] for t in user_only] == ["user", "user"]


def test_export_embeddings_applies_strip_document(tmp_path, monkeypatch):
    seen: list[list[dict]] = []

    class FakeModel:
        def encode(self, sessions):
            seen.extend(sessions)
            return torch.tensor([[float(len(session))] for session in sessions])

    monkeypatch.setattr(export, "load_checkpoint", lambda checkpoint, device: FakeModel())
    source = tmp_path / "input.jsonl"
    source.write_text(
        json.dumps(
            {
                "session_id": "s1",
                "real_conversation": [
                    {"role": "user", "content": "edit"},
                    {
                        "role": "assistant",
                        "content": (
                            "Sure, here's an update.\n\n"
                            "Subject: Hello\n\nBody of the long document draft goes here "
                            "with enough words to look like a real email body rather than meta."
                        ),
                    },
                    {"role": "user", "content": "shorter"},
                ],
            }
        )
        + "\n",
        encoding="utf-8",
    )
    output = tmp_path / "out.jsonl"
    export.export_embeddings(
        "checkpoint",
        source,
        output,
        session_preprocess="strip_document",
    )
    assert len(seen) == 1
    asst = [t for t in seen[0] if t["role"] == "assistant"][0]
    assert "Sure" in asst["content"]
    assert "Subject:" not in asst["content"]


def test_normalize_session_preprocess_rejects_unknown():
    with pytest.raises(ValueError, match="session_preprocess"):
        preprocess_turns([], "nope")
