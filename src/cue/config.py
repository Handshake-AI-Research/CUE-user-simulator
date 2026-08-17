"""TOML configuration for the production CUE runner."""

from __future__ import annotations

import tomllib
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

_CONVERSATION_REQUIRED = "conditioned mode requires conversation_path"
_CONVERSATION_CONFLICT = "conversation_path is only valid in conditioned mode"
_INVALID_N = "n must be at least 1"
_SIMULATOR_FIELDS = "simulator_backend, simulator_model, and scenario must be set together"


@dataclass(frozen=True)
class CueConfig:
    """Configuration loaded by ``cue --config``.

    A run produces either a user-conditioned CUE from ``conversation_path`` or a
    sampled CUE from the prior. Simulator fields are optional: omit them when the
    application only needs the manual/steering prompt.
    """

    mode: Literal["conditioned", "sampled"]
    output_dir: Path
    model: str = "handshake-ai-research/cue"
    device: str = "cpu"
    conversation_path: Path | None = None
    session_preprocess: Literal["full", "strip_document", "user_only"] = "full"
    seed: int = 0
    n: int = 1
    example_pool: str | None = None
    example_retrieval: bool = False
    simulator_backend: Literal["openai", "hf"] | None = None
    simulator_model: str | None = None
    scenario: str | None = None
    history_path: Path | None = None
    api_key_env: str | None = None
    base_url: str | None = None

    @classmethod
    def from_toml(cls, path: str | Path) -> CueConfig:
        """Load and validate a runner config."""

        config_path = Path(path).expanduser().resolve()
        with config_path.open("rb") as handle:
            raw = tomllib.load(handle)
        root = Path.cwd()
        for key in ("conversation_path", "history_path", "output_dir"):
            value = raw.get(key)
            if value is not None:
                candidate = Path(str(value)).expanduser()
                raw[key] = candidate if candidate.is_absolute() else root / candidate
        config = cls(**raw)
        config._validate()
        return config

    def _validate(self) -> None:
        if self.mode == "conditioned" and self.conversation_path is None:
            raise ValueError(_CONVERSATION_REQUIRED)
        if self.mode == "sampled" and self.conversation_path is not None:
            raise ValueError(_CONVERSATION_CONFLICT)
        if self.n < 1:
            raise ValueError(_INVALID_N)
        simulator_fields = (
            self.simulator_backend,
            self.simulator_model,
            self.scenario,
        )
        if any(value is not None for value in simulator_fields) and not all(
            value is not None for value in simulator_fields
        ):
            raise ValueError(_SIMULATOR_FIELDS)

    @property
    def runs_simulator(self) -> bool:
        return self.simulator_backend is not None

    def public_dict(self) -> dict[str, Any]:
        """Serializable config without secret values."""

        return {key: str(value) if isinstance(value, Path) else value for key, value in vars(self).items()}
