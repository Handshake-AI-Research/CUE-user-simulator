"""UserLM baseline registration (inference-only)."""

from __future__ import annotations

from cue_training.baselines.common.config import RolloutConfig
from cue_training.baselines.common.simulator import Baseline, UserSimulator
from cue_training.runlog.log import log


class UserLMBaseline(Baseline):
    name = "userlm"

    def load(self, cfg: RolloutConfig) -> UserSimulator:
        # vLLM fast path: UserLM-8b is served by a local vLLM OpenAI server, so generation is
        # batched (high throughput) instead of single-stream HF. Enabled by threading
        # userlm_vllm_base_url/userlm_vllm_model through cfg.extra. The paper's decoding
        # guardrails are approximated via vLLM sampling params (see UserLMVLLMSimulator).
        vllm_base_url = cfg.extra.get("userlm_vllm_base_url")
        if vllm_base_url:
            from cue_training.baselines.userlm.simulator import UserLMVLLMSimulator

            model = cfg.extra.get("userlm_vllm_model") or "userlm"
            log("userlm", f"serving UserLM via vLLM ({vllm_base_url}, model={model}).")
            return UserLMVLLMSimulator(
                base_url=str(vllm_base_url),
                model=str(model),
                api_key_env=cfg.extra.get("userlm_vllm_api_key_env", "HOSTED_VLLM_API_KEY"),
            )

        from cue_training.baselines.userlm.simulator import UserLMSimulator

        model_id = cfg.extra.get("model_id", "microsoft/UserLM-8b")
        return UserLMSimulator(model_id=model_id, device=cfg.device)
