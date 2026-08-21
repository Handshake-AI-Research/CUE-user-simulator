"""USP baseline: published HuggingFace model only (no local training)."""

from __future__ import annotations

from cue_training.baselines.common.config import RolloutConfig, TrainConfig
from cue_training.baselines.common.simulator import Baseline, UserSimulator
from cue_training.baselines.usp.extractor import PROFILES_FILE, load_profiles
from cue_training.runlog.log import log, warn

DEFAULT_USP_HF_MODEL = "wangkevin02/USP"


class USPBaseline(Baseline):
    name = "usp"

    def train(self, cfg: TrainConfig) -> None:
        log(
            "usp",
            f"USP uses the published HF checkpoint ({DEFAULT_USP_HF_MODEL}); "
            "local SFT/RLCC training is not supported in this tree.",
        )

    def load(self, cfg: RolloutConfig) -> UserSimulator:
        artifacts = cfg.artifacts_path(self.name)
        profiles = load_profiles(artifacts / PROFILES_FILE)
        if not profiles:
            warn(
                "usp",
                f"no {PROFILES_FILE} in {artifacts}; using a default profile. "
                "Tau2/SimArena persona prep supplies per-episode profiles at rollout.",
            )

        vllm_base_url = (
            cfg.extra.get("usp_official_vllm_base_url")
            or cfg.extra.get("usp_vllm_base_url")
        )
        if not vllm_base_url:
            raise RuntimeError(
                "USP requires a vLLM server for the published HF model. "
                "Pass usp_vllm_base_url (or usp_official_vllm_base_url) pointing at "
                f"a `vllm serve {DEFAULT_USP_HF_MODEL}` endpoint."
            )

        from cue_training.baselines.usp.simulator import USPOfficialVLLMSimulator

        model = (
            cfg.extra.get("usp_official_vllm_model")
            or cfg.extra.get("usp_vllm_model")
            or DEFAULT_USP_HF_MODEL
        )
        log("usp", f"serving published USP via vLLM ({vllm_base_url}, model={model}).")
        return USPOfficialVLLMSimulator(
            profiles,
            base_url=str(vllm_base_url),
            model=str(model),
            api_key_env=cfg.extra.get(
                "usp_official_vllm_api_key_env",
                cfg.extra.get("usp_vllm_api_key_env", "HOSTED_VLLM_API_KEY"),
            ),
            temperature=cfg.temperature,
        )
