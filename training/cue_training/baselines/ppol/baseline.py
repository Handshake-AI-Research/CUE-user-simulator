"""PPol baseline (faithful): evolved persona generator G(c, D, N) via OpenEvolve.

Training (OpenEvolve program search) is heavy and tau2-coupled: the rollout
orchestrator trains once per simulator model with tau2 fitness
(``scripts/train_ppol.sh`` → frozen ``best_program.py`` under
``ppol/retail_airline_<simtag>/``). At rollout, personas are generated from that
program on tau2 tasks and reused as a pool for other domains; the simulator
here just consumes the injected ``_persona``.
"""

from __future__ import annotations

from cue_training.baselines.common.config import RolloutConfig, TrainConfig
from cue_training.baselines.common.simulator import Baseline, UserSimulator
from cue_training.runlog.log import log


class PPolBaseline(Baseline):
    name = "ppol"

    def train(self, cfg: TrainConfig) -> None:  # noqa: ARG002
        log(
            self.name,
            "PPol training is the OpenEvolve program search; run it explicitly via "
            "scripts/train_ppol.sh (produces a frozen best_program.py). Skipping here.",
        )

    def load(self, cfg: RolloutConfig) -> UserSimulator:
        from cue_training.baselines.ppol.simulator import PPolSimulator

        # Personas are injected per episode by the tau2 persona sidecar (into ``_persona``);
        # an empty pool falls back to the default persona if no sidecar row is supplied.
        return PPolSimulator(
            [],
            model=cfg.sim_model,
            api_key_env=cfg.sim_api_key_env,
            api_base=cfg.sim_api_base,
            temperature=cfg.temperature,
            max_tokens=cfg.sim_max_tokens,
        )
