"""Simulator protocol and helpers."""

from __future__ import annotations

from typing import TYPE_CHECKING, Protocol

if TYPE_CHECKING:
    from collections.abc import Sequence

from cue.manual import PersonaManual
from cue.steering import user_sim_system


class Simulator(Protocol):
    """Any chat model that can role-play a user (or the assistant)."""

    def chat(
        self,
        messages: Sequence[dict[str, str]],
        *,
        system: str | None = None,
        temperature: float = 0.8,
        max_tokens: int = 160,
    ) -> str: ...


def run_user_turn(
    simulator: Simulator,
    *,
    scenario: str,
    history: Sequence[dict[str, str]],
    manual: PersonaManual | str | None = None,
    temperature: float = 0.8,
    max_tokens: int = 160,
) -> str:
    """Produce the next user utterance, optionally steered by a CUE manual.

    ``history`` is the conversation from the *assistant's* point of view
    (user/assistant roles as usual). Internally roles are flipped so the
    simulator sees its own past turns as assistant messages.
    """

    steering = manual.steering_prompt if isinstance(manual, PersonaManual) else manual
    system = user_sim_system(scenario, steering)
    flipped = [
        {
            "role": "assistant" if turn["role"] == "user" else "user",
            "content": turn["content"],
        }
        for turn in history
    ]
    return simulator.chat(flipped, system=system, temperature=temperature, max_tokens=max_tokens)
