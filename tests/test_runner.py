"""Stable config-runner output tests; model loading is mocked."""

from __future__ import annotations

import json
from typing import TYPE_CHECKING, Any
from unittest.mock import patch

from cue.config import CueConfig
from cue.manual import PersonaManual
from cue.runner import run

if TYPE_CHECKING:
    from pathlib import Path


class _FakeCue:
    def sample(self, **_: Any) -> PersonaManual:
        return PersonaManual(
            commands=["Keep replies brief."],
            steering_prompt="Behavior manual:\n- Keep replies brief.",
            source="sampled",
        )


def test_runner_writes_stable_artifacts(tmp_path: Path) -> None:
    config = CueConfig(mode="sampled", output_dir=tmp_path, n=1)

    with patch("cue.runner.Cue.from_pretrained", return_value=_FakeCue()):
        info = run(config)

    manuals = json.loads((tmp_path / "manuals.json").read_text(encoding="utf-8"))
    assert manuals[0]["source"] == "sampled"
    assert manuals[0]["commands"] == ["Keep replies brief."]
    assert (tmp_path / "steering_prompts.txt").read_text(encoding="utf-8").startswith("Behavior manual:")
    assert (tmp_path / "info.json").exists()
    assert info["manual_count"] == 1
    assert not (tmp_path / "simulated_users.json").exists()
