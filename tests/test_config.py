"""Config contract tests; no model download required."""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

from cue.config import CueConfig

if TYPE_CHECKING:
    from pathlib import Path


def test_loads_conditioned_config(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.chdir(tmp_path)
    (tmp_path / "conversation.json").write_text("[]", encoding="utf-8")
    path = tmp_path / "cue.toml"
    path.write_text(
        'mode = "conditioned"\nconversation_path = "conversation.json"\noutput_dir = "output"\n',
        encoding="utf-8",
    )

    config = CueConfig.from_toml(path)

    assert config.model == "handshake-ai-research/cue"
    assert config.conversation_path == tmp_path / "conversation.json"
    assert config.output_dir == tmp_path / "output"
    assert not config.runs_simulator


def test_conditioned_mode_requires_conversation(tmp_path: Path) -> None:
    path = tmp_path / "cue.toml"
    path.write_text('mode = "conditioned"\noutput_dir = "output"\n', encoding="utf-8")

    with pytest.raises(ValueError, match="conversation_path"):
        CueConfig.from_toml(path)


def test_simulator_fields_are_all_or_nothing(tmp_path: Path) -> None:
    path = tmp_path / "cue.toml"
    path.write_text(
        'mode = "sampled"\noutput_dir = "output"\nsimulator_backend = "hf"\n',
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="must be set together"):
        CueConfig.from_toml(path)
