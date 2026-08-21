"""Shared per-episode cloning helpers for prompt-based simulators."""

from __future__ import annotations

import copy
import random
from typing import Any


class SeededSpawnMixin:
    """Clone a simulator and reseed its `_rng` for one rollout episode."""

    _seed: int
    _rng: random.Random

    def spawn(self, episode_index: int) -> Any:
        clone = copy.copy(self)
        clone._rng = random.Random(self._seed + episode_index)
        return clone
