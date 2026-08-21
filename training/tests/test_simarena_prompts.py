from __future__ import annotations

import pytest

from cue_training.evaluation.simulatorarena.prompts import native_user_prompt, parse_native_user_output


def _require_prompt_templates() -> None:
    from cue_training.evaluation.simulatorarena import prompts as P

    if not P._PROMPT_ROOT.is_dir():
        pytest.skip("SimulatorArena prompt templates are not vendored; clone the external tree")


def test_writing_uses_native_initial_and_followup_templates():
    _require_prompt_templates()
    initial = native_user_prompt(
        domain="writing",
        history=[],
        user_profile="PROFILE",
        task="fallback",
        document_type="email",
        intent="thank my sister",
        background="She helped with the baby.",
    )
    assert "realistic and appropriate **initial message**" in initial
    assert "Document Type: email" in initial
    assert "PROFILE" in initial
    assert "{user_profile}" not in initial

    followup = native_user_prompt(
        domain="writing",
        history=[
            {"role": "user", "content": "Please draft it."},
            {"role": "assistant", "content": "Here is a draft."},
        ],
        user_profile="PROFILE",
        task="fallback",
        document_type="email",
        intent="thank my sister",
        background="notes",
    )
    assert "- You: Please draft it." in followup
    assert "- AI Writing Assistant: Here is a draft." in followup
    assert 'output only "terminate conversation"' in followup


def test_math_uses_native_profile_template():
    _require_prompt_templates()
    prompt = native_user_prompt(
        domain="math",
        history=[{"role": "assistant", "content": "What have you tried?"}],
        user_profile="PROFILE",
        task="fallback",
        math_problem="Solve x + 2 = 5.",
    )
    assert "role-playing as a student" in prompt
    assert "Solve x + 2 = 5." in prompt
    assert "- AI Tutor: What have you tried?" in prompt
    assert "PROFILE" in prompt


def test_native_output_parsing():
    assert parse_native_user_output("Thought: revise it\n\nMessage: Make it warmer.") == (
        "Make it warmer.",
        False,
    )
    assert parse_native_user_output("Thought: done\n\nResponse: terminate conversation") == ("", True)
