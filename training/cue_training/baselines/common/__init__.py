"""Shared harness for the user-simulator baselines."""

from cue_training.baselines.common.config import RolloutConfig, TrainConfig
from cue_training.baselines.common.domains import DOMAINS, Domain, normalized_path, resolve_domains
from cue_training.baselines.common.simulator import (
    Baseline,
    BaseUserSimulator,
    PromptUserSimulator,
    UserSimulator,
)

__all__ = [
    "RolloutConfig",
    "TrainConfig",
    "DOMAINS",
    "Domain",
    "normalized_path",
    "resolve_domains",
    "Baseline",
    "BaseUserSimulator",
    "PromptUserSimulator",
    "UserSimulator",
]
