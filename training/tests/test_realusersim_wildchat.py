"""Unit tests for RealUserSim WildChat curation (paper Appendix A.1 stages 1–2)."""

from __future__ import annotations

import json

from cue_training.baselines.realusersim.format_persona import format_persona_block
from cue_training.baselines.realusersim.profiles import normalize_profile
from cue_training.baselines.realusersim.wildchat_curate import (
    curate_wildchat,
    filter_row,
    group_user_transcripts,
    is_gpt4o_model,
    trim_greeting_thanks,
)


def test_is_gpt4o_excludes_mini():
    assert is_gpt4o_model("gpt-4o")
    assert is_gpt4o_model("gpt-4o-2024-05-13")
    assert not is_gpt4o_model("gpt-4o-mini")
    assert not is_gpt4o_model("gpt-4")


def test_trim_greeting_thanks_bookends():
    msgs = [
        {"role": "user", "content": "hi"},
        {"role": "assistant", "content": "Hello! How can I help?"},
        {"role": "user", "content": "explain quantum tunneling briefly"},
        {"role": "assistant", "content": "Sure — here is a short explanation..."},
        {"role": "user", "content": "thanks"},
    ]
    trimmed = trim_greeting_thanks(msgs)
    assert trimmed[0]["role"] == "assistant"
    assert any(t["content"].startswith("explain") for t in trimmed)
    assert trimmed[-1]["role"] == "assistant"
    assert not any(t["content"].lower() in {"hi", "thanks"} for t in trimmed)


def test_filter_row_paper_gates(tmp_path):
    good = {
        "conversation_hash": "abc",
        "model": "gpt-4o-2024-08-06",
        "hashed_ip": "user-1",
        "toxic": False,
        "conversation": [
            {"role": "user", "content": "hi", "language": "English"},
            {"role": "assistant", "content": "Hello there, how can I help you today?"},
            {"role": "user", "content": "write a python function to reverse a list", "language": "English"},
            {"role": "assistant", "content": "def reverse(xs): return xs[::-1]"},
            {"role": "user", "content": "add type hints please", "language": "English"},
            {"role": "assistant", "content": "def reverse(xs: list) -> list: return xs[::-1]"},
        ],
    }
    rec = filter_row(good)
    assert rec is not None
    assert rec["hashed_ip"] == "user-1"
    assert rec["n_turns"] >= 3

    bad_model = dict(good, model="gpt-3.5-turbo")
    assert filter_row(bad_model) is None

    short = dict(
        good,
        conversation=[
            {"role": "user", "content": "hi", "language": "English"},
            {"role": "assistant", "content": "hey"},
        ],
    )
    assert filter_row(short) is None


def test_curate_and_group_users(tmp_path):
    rows = [
        {
            "conversation_hash": "c1",
            "model": "gpt-4o",
            "hashed_ip": "u1",
            "toxic": False,
            "conversation": [
                {"role": "user", "content": "msg one from user", "language": "en"},
                {"role": "assistant", "content": "reply one"},
                {"role": "user", "content": "msg two from user", "language": "en"},
                {"role": "assistant", "content": "reply two"},
            ],
        },
        {
            "conversation_hash": "c2",
            "model": "gpt-4o",
            "hashed_ip": "u1",
            "toxic": False,
            "conversation": [
                {"role": "user", "content": "another session turn", "language": "en"},
                {"role": "assistant", "content": "ok"},
                {"role": "user", "content": "and a followup", "language": "en"},
                {"role": "assistant", "content": "done"},
            ],
        },
        {
            "conversation_hash": "c3",
            "model": "gpt-4",
            "hashed_ip": "u2",
            "toxic": False,
            "conversation": [
                {"role": "user", "content": "should drop", "language": "en"},
                {"role": "assistant", "content": "no"},
                {"role": "user", "content": "still drop", "language": "en"},
            ],
        },
    ]
    path, stats = curate_wildchat(tmp_path, rows=rows)
    assert stats["kept_trajs"] == 2
    assert stats["unique_users"] == 1
    by_user = group_user_transcripts(path)
    assert set(by_user) == {"u1"}
    assert "msg one from user" in by_user["u1"]
    assert "another session turn" in by_user["u1"]


def test_import_published_profiles_verbatim(tmp_path, monkeypatch):
    import cue_training.baselines.realusersim.profiles as profiles_mod

    src = tmp_path / "consolidated_user_profiles.jsonl"
    row = {
        "user_ip": "hash1",
        "demographics": {"age": {"value": "25-34", "source": "inferred", "confidence": 0.7}},
        "linguistic_profile": "Command: Use lowercase.\nExamples: \"ok\"",
        "num_conversations": 3,
        "profile_completeness": 0.571,
        "conversations_used": ["h1", "h2"],
    }
    src.write_text(json.dumps(row) + "\n", encoding="utf-8")
    monkeypatch.setattr(profiles_mod, "hf_hub_download", lambda **kw: str(src), raising=False)

    out = profiles_mod.import_published_profiles(tmp_path / "artifacts")
    written = [json.loads(l) for l in out.read_text().splitlines() if l.strip()]
    assert len(written) == 1
    got = written[0]
    # Every published field preserved verbatim.
    for key, value in row.items():
        assert got[key] == value
    # manual_sections mirrors linguistic_profile for the formatter.
    assert got["manual_sections"] == row["linguistic_profile"]


def test_format_published_linguistic_profile():
    profile = {
        "demographics": {
            "age": {"value": "18-24", "source": "inferred", "confidence": 0.6},
            "occupation": {"value": None, "source": None, "confidence": None},
        },
        "linguistic_profile": 'Command: Use lowercase.\nExamples: "ok sure"',
    }
    block = format_persona_block(profile)
    assert "Age: 18-24" in block
    assert "Use lowercase" in block
    norm = normalize_profile(
        {
            "demographics": profile["demographics"],
            "linguistic_profile": profile["linguistic_profile"],
            "command_examples": [],
        }
    )
    assert norm is not None
    assert "manual_sections" in norm
