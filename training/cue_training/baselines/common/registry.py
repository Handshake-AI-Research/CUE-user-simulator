"""Lazy registry mapping baseline names to their ``Baseline`` implementations."""

from __future__ import annotations

from cue_training.baselines.common.simulator import Baseline

BASELINE_NAMES = [
    "userlm",
    "usp",
    "ppol",
    "realusersim",
]


def get_baseline(name: str) -> Baseline:
    key = name.strip().lower().replace("-", "_")
    if key == "userlm":
        from cue_training.baselines.userlm.baseline import UserLMBaseline

        return UserLMBaseline()
    if key == "usp":
        from cue_training.baselines.usp.baseline import USPBaseline

        return USPBaseline()
    if key == "ppol":
        from cue_training.baselines.ppol.baseline import PPolBaseline

        return PPolBaseline()
    if key in ("realusersim", "realusersim_paired_noex"):
        from cue_training.baselines.realusersim.baseline import RealUserSimBaseline

        return RealUserSimBaseline()
    raise ValueError(
        f"Unknown baseline {name!r}. Choose from: {', '.join(BASELINE_NAMES)} "
        "(or realusersim_paired_noex)"
    )
