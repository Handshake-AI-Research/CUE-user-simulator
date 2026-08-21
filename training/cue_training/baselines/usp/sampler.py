"""Diverse implicit-profile sampler (USP)."""

from __future__ import annotations

import random

from cue_training.baselines.usp.extractor import profile_text

_DEFAULT_PROFILE = {
    "goals": "accomplish the task efficiently",
    "knowledge_level": "intermediate",
    "preferences": "concise answers",
    "personality": "pragmatic",
    "communication_style": "brief and direct",
}


class ProfileSampler:
    """Samples implicit profiles, lightly biased toward unseen styles for diversity."""

    def __init__(self, profiles: list[dict], seed: int = 0) -> None:
        self.profiles = profiles or [_DEFAULT_PROFILE]
        self._rng = random.Random(seed)
        self._seen_styles: set[str] = set()

    def sample(self) -> dict:
        unseen = [
            p
            for p in self.profiles
            if str(p.get("communication_style", "")) not in self._seen_styles
        ]
        pool = unseen or self.profiles
        chosen = self._rng.choice(pool)
        self._seen_styles.add(str(chosen.get("communication_style", "")))
        if len(self._seen_styles) >= len(self.profiles):
            self._seen_styles.clear()
        return chosen

    def sample_text(self) -> str:
        return profile_text(self.sample())
