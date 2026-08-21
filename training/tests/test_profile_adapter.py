import pytest

from cue_training.data.profile_adapter import to_profile
from cue_training.data.schema import ManualValidationError


def test_adapts_legacy_profile_and_limits_unique_examples():
    profile = to_profile(
        {
            "commands": [
                {
                    "command": "Be concise.",
                    "examples": ["ok", "same", "same", "third"],
                },
                {"text": "Ask directly.", "examples": ["fourth", "fifth"]},
            ]
        }
    )
    assert profile == {
        "commands": ["Be concise.", "Ask directly."],
        "examples": ["ok", "same", "third", "fourth"],
    }


def test_adapts_unified_profile():
    assert to_profile(
        {
            "commands": ["Be concise.", "Be concise.", "Ask directly."],
            "examples": ["one", "two"],
        }
    ) == {
        "commands": ["Be concise.", "Ask directly."],
        "examples": ["one", "two"],
    }


def test_rejects_profile_without_commands():
    with pytest.raises(ManualValidationError, match="commands"):
        to_profile({"commands": []})
