"""RealUserSim baseline: extract persona pool (train), ground simulator (rollout)."""

from __future__ import annotations

from cue_training.baselines.common.config import RolloutConfig, TrainConfig
from cue_training.baselines.common.simulator import Baseline, UserSimulator
from cue_training.baselines.realusersim.profiles import PROFILES_FILE, extract_profiles, load_profiles
from cue_training.runlog.log import warn


class RealUserSimBaseline(Baseline):
    name = "realusersim"

    def train(self, cfg: TrainConfig) -> None:
        extract_profiles(cfg)

    def load(self, cfg: RolloutConfig) -> UserSimulator:
        from cue_training.baselines.realusersim.simulator import RealUserSimSimulator

        artifacts = cfg.artifacts_path(self.name)
        profiles = load_profiles(artifacts / PROFILES_FILE)
        if not profiles:
            warn(
                "realusersim",
                f"no {PROFILES_FILE} found in {artifacts}; "
                "using a default persona. Run `train` first for grounded profiles.",
            )
        return RealUserSimSimulator(
            profiles,
            model=cfg.sim_model,
            api_key_env=cfg.sim_api_key_env,
            api_base=cfg.sim_api_base,
            temperature=cfg.temperature,
            max_tokens=cfg.sim_max_tokens,
        )
