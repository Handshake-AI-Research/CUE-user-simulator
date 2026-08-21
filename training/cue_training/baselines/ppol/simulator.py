"""PPol simulator: a base LLM user simulator conditioned on an evolved persona."""

from __future__ import annotations

import random
from typing import Any

from cue_training.baselines.common.prompts import user_system_prompt, user_turn_prompt
from cue_training.baselines.common.spawn import SeededSpawnMixin
from cue_training.baselines.common.simulator import PromptUserSimulator
from cue_training.baselines.ppol.persona_prompt import inject_persona_into_system_prompt

_DEFAULT_PERSONA = (
    "You communicate with these traits:\n"
    "- Message length: terse\n- Register: casual\n- How you phrase requests: direct\n"
    "- Spelling/grammar care: occasional typos"
)


class PPolSimulator(SeededSpawnMixin, PromptUserSimulator):
    def __init__(self, personas: list[dict], *, seed: int = 0, **kwargs: Any) -> None:
        super().__init__("ppol", **kwargs)
        self.personas = personas
        self._seed = seed
        self._rng = random.Random(seed)
        self._persona = _DEFAULT_PERSONA
        # Fitness-weighted sampling weights.
        self._weights = [max(1e-3, float(p.get("fitness", 0.5))) for p in personas]

    def _on_episode_start(self, task: str, metadata: dict[str, Any]) -> None:  # noqa: ARG002
        if self.personas:
            chosen = self._rng.choices(self.personas, weights=self._weights, k=1)[0]
            self._persona = chosen.get("persona", _DEFAULT_PERSONA)

    def augmentation(self, task: str, metadata: dict[str, Any]) -> str:  # noqa: ARG002
        # Persona is applied via PERSONA_INJECTION_TEMPLATE in _messages (train/rollout parity).
        return ""

    def _messages(
        self, task: str, history: list[dict[str, str]], metadata: dict[str, Any]
    ) -> list[dict[str, str]]:
        # Match train fitness: native(+task-block) system from the harness, then the same
        # PERSONA_INJECTION_TEMPLATE wrapper. Do NOT also stack CUE user_system_prompt.
        if self.extra_system_prompt:
            system = self.extra_system_prompt.strip()
            if self._persona:
                system = inject_persona_into_system_prompt(system, self._persona)
        else:
            system = user_system_prompt(task, metadata, self._persona)
        return [
            {"role": "system", "content": system},
            {"role": "user", "content": user_turn_prompt(task, history)},
        ]
