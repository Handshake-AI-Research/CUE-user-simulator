"""Unit tests that do not need Hub weights."""

from __future__ import annotations

import json
from typing import TYPE_CHECKING

from cue.conversation import load_conversation, normalize_messages
from cue.manual import PersonaManual
from cue.steering import render_steering, user_sim_system

if TYPE_CHECKING:
    from pathlib import Path


def test_normalize_openai_messages() -> None:
    turns = normalize_messages(
        [
            {"role": "user", "content": "hi"},
            {"role": "assistant", "content": "hello"},
            {"role": "human", "content": "more"},
        ]
    )
    assert [t["role"] for t in turns] == ["user", "assistant", "user"]


def test_normalize_transcript() -> None:
    turns = normalize_messages("user: can you tighten this\nassistant: Sure.\nuser: still too long")
    assert len(turns) == 3
    assert turns[0]["content"] == "can you tighten this"


def test_load_conversation_json(tmp_path: Path) -> None:
    path = tmp_path / "c.json"
    path.write_text(
        json.dumps([{"role": "user", "content": "a"}, {"role": "assistant", "content": "b"}]),
        encoding="utf-8",
    )
    assert load_conversation(str(path))[0]["content"] == "a"


def test_render_steering_sections() -> None:
    text = render_steering(
        general=["be brief"],
        specific=["ask about price"],
        style=["lowercase"],
        examples=["ok thanks"],
    )
    assert "General behavior:" in text
    assert "User-specific behavior:" in text
    assert "Writing style:" in text
    assert "ok thanks" in text


def test_persona_manual_system_addon() -> None:
    manual = PersonaManual(commands=["x"], steering_prompt="- be brief")
    assert "Behave like this specific user" in manual.as_system_addon()
    assert user_sim_system("buy a ticket", manual.steering_prompt).endswith("- be brief")
