"""Protocols and base classes for baselines and their user simulators."""

from __future__ import annotations

import asyncio
from abc import ABC, abstractmethod
from collections.abc import Callable
from typing import Any, Protocol, runtime_checkable

from cue_training.baselines.common.config import RolloutConfig, TrainConfig
from cue_training.baselines.common.llm import achat, chat
from cue_training.baselines.common.prompts import user_system_prompt, user_turn_prompt

Turn = dict[str, str]


@runtime_checkable
class UserSimulator(Protocol):
    """Produces user turns inside the closed-loop rollout driver."""

    def first_turn(self, task: str, metadata: dict[str, Any]) -> str: ...

    def next_turn(
        self, task: str, history: list[Turn], metadata: dict[str, Any]
    ) -> str: ...

    async def first_turn_async(self, task: str, metadata: dict[str, Any]) -> str: ...

    async def next_turn_async(
        self, task: str, history: list[Turn], metadata: dict[str, Any]
    ) -> str: ...

    def is_done(self, history: list[Turn]) -> bool: ...


class BaseUserSimulator:
    """Convenience base with a no-op ``is_done`` and shared first/next plumbing."""

    name: str = "base"

    def _on_episode_start(self, task: str, metadata: dict[str, Any]) -> None:  # noqa: ARG002
        """Per-episode setup hook (e.g. persona/profile selection).

        Runs once before the first user turn, shared by the sync and async
        first-turn paths so subclasses only implement selection logic once.
        """

    def first_turn(self, task: str, metadata: dict[str, Any]) -> str:
        self._on_episode_start(task, metadata)
        return self.next_turn(task, [], metadata)

    def next_turn(
        self, task: str, history: list[Turn], metadata: dict[str, Any]
    ) -> str:
        raise NotImplementedError

    async def first_turn_async(self, task: str, metadata: dict[str, Any]) -> str:
        self._on_episode_start(task, metadata)
        return await self.next_turn_async(task, [], metadata)

    async def next_turn_async(
        self, task: str, history: list[Turn], metadata: dict[str, Any]
    ) -> str:
        """Async user turn. Defaults to offloading the sync path to a thread.

        Prompt-based simulators override this with a native ``acompletion`` call;
        non-API simulators (e.g. GPU-served) inherit this thread fallback so the
        async rollout driver still works for them.
        """

        return await asyncio.to_thread(self.next_turn, task, history, metadata)

    def is_done(self, history: list[Turn]) -> bool:  # noqa: ARG002
        return False

    def spawn(self, episode_index: int) -> "BaseUserSimulator":  # noqa: ARG002
        """Return an instance safe to run one episode on its own thread.

        Default returns ``self`` (stateless / GPU simulators are run
        sequentially). Simulators with per-episode mutable state override this
        to hand back an isolated copy for the parallel rollout driver.
        """

        return self


class PromptUserSimulator(BaseUserSimulator):
    """An LLM user simulator driven by a system prompt + textual transcript.

    Trainable/profile-based baselines override :meth:`augmentation` to inject
    persona policies, implicit profiles, or persona manuals into the prompt.
    """

    def __init__(
        self,
        name: str,
        *,
        model: str = "gpt-4o-mini",
        api_key_env: str | None = "OPENAI_API_KEY",
        api_base: str | None = None,
        temperature: float = 0.8,
        max_tokens: int = 512,
    ) -> None:
        self.name = name
        self.model = model
        self.api_key_env = api_key_env
        self.api_base = api_base
        self.temperature = temperature
        self.max_tokens = max_tokens
        # Optional harness-supplied system prompt prepended for cross-model comparability
        # (e.g. tau2's standard user-simulator prompt). Empty by default (no change).
        self.extra_system_prompt: str = ""
        # Optional harness-supplied (task, history, persona) -> messages builder. When set it
        # replaces the default system+turn prompt so a baseline can be run in a shared format
        # (e.g. SimulatorArena's native CoT template) instead of its own. None = no change.
        self.prompt_builder: Callable[[str, list[Turn], str], list[dict[str, str]]] | None = None

    def augmentation(self, task: str, metadata: dict[str, Any]) -> str:  # noqa: ARG002
        return ""

    def _messages(
        self, task: str, history: list[Turn], metadata: dict[str, Any]
    ) -> list[dict[str, str]]:
        if self.prompt_builder is not None:
            return self.prompt_builder(task, history, self.augmentation(task, metadata))
        system = user_system_prompt(task, metadata, self.augmentation(task, metadata))
        if self.extra_system_prompt:
            system = f"{self.extra_system_prompt.strip()}\n\n{system}"
        return [
            {"role": "system", "content": system},
            {"role": "user", "content": user_turn_prompt(task, history)},
        ]

    def next_turn(
        self, task: str, history: list[Turn], metadata: dict[str, Any]
    ) -> str:
        return chat(
            self._messages(task, history, metadata),
            model=self.model,
            api_key_env=self.api_key_env,
            api_base=self.api_base,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
        )

    async def next_turn_async(
        self, task: str, history: list[Turn], metadata: dict[str, Any]
    ) -> str:
        return await achat(
            self._messages(task, history, metadata),
            model=self.model,
            api_key_env=self.api_key_env,
            api_base=self.api_base,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
        )


class Baseline(ABC):
    """A baseline that can optionally ``train`` and must ``load`` a simulator."""

    name: str = "baseline"

    def train(self, cfg: TrainConfig) -> None:  # noqa: ARG002
        from cue_training.runlog.log import log

        log(self.name, "no training required (inference-only baseline).")

    @abstractmethod
    def load(self, cfg: RolloutConfig) -> UserSimulator:
        raise NotImplementedError
