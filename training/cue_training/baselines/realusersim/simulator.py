"""RealUserSim simulator: an LLM grounded in a WildChat-sampled persona profile."""

from __future__ import annotations

import random
from typing import Any

from cue_training.baselines.common.spawn import SeededSpawnMixin
from cue_training.baselines.common.simulator import PromptUserSimulator
from cue_training.baselines.realusersim.format_persona import ANTI_NORMALIZATION, format_persona_block

Turn = dict[str, str]

# Back-compat alias for imports that still use the private name.
_ANTI_NORMALIZATION = ANTI_NORMALIZATION

_DEFAULT_PROFILE = {
    "demographics": {
        "age": {"value": "25-34", "source": "inferred", "confidence": 0.5},
        "gender": {"value": None, "source": None, "confidence": None},
        "education": {"value": None, "source": None, "confidence": None},
        "occupation": {"value": "everyday user", "source": "inferred", "confidence": 0.4},
        "location": {"value": None, "source": None, "confidence": None},
        "income": {"value": None, "source": None, "confidence": None},
        "marital_status": {"value": None, "source": None, "confidence": None},
        "nationality": {"value": None, "source": None, "confidence": None},
    },
    "additional_background": "Experienced with customer service chat interactions.",
    "command_examples": [
        {
            "command": "Be terse and get straight to the point.",
            "examples": ["can you fix this?", "still broken"],
        },
        {
            "command": "Skip pleasantries and do not over-explain.",
            "examples": ["ok now add the tests"],
        },
        {
            "command": "Push back or clarify when the assistant misunderstands.",
            "examples": ["no thats not what i asked"],
        },
    ],
}


class RealUserSimSimulator(SeededSpawnMixin, PromptUserSimulator):
    def __init__(self, profiles: list[dict], *, seed: int = 0, **kwargs: Any) -> None:
        super().__init__("realusersim", **kwargs)
        self.profiles = profiles or [_DEFAULT_PROFILE]
        self._seed = seed
        self._rng = random.Random(seed)
        self._profile = self.profiles[0]

    def _on_episode_start(self, task: str, metadata: dict[str, Any]) -> None:  # noqa: ARG002
        self._profile = self._rng.choice(self.profiles)

    def augmentation(self, task: str, metadata: dict[str, Any]) -> str:  # noqa: ARG002
        profile = self._profile
        # An injected persona may be a prose manual (e.g. SimulatorArena's extracted profile
        # text) rather than the distilled dict; use it as-is.
        if isinstance(profile, str):
            block = profile.strip()
        else:
            include_examples = True
            if isinstance(profile, dict):
                include_examples = profile.get("include_examples", True) is not False
            block = format_persona_block(profile, include_examples=include_examples)
        return (
            "You are imitating ONE specific real user with this persona.\n"
            f"{block}\n\n"
            f"{ANTI_NORMALIZATION}"
        )
