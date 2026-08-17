"""Config-driven CUE application runner."""

from __future__ import annotations

import json
import os
from typing import TYPE_CHECKING, Any

from cue.conversation import load_conversation
from cue.runtime import Cue
from cue.simulator import (
    HuggingFaceSimulator,
    OpenAICompatSimulator,
    run_user_turn,
)

if TYPE_CHECKING:
    from cue.config import CueConfig

_INVALID_BACKEND = "unsupported simulator backend"
_INVALID_CONDITIONED = "validated conditioned config has no conversation_path"
_INVALID_SIMULATOR = "validated simulator config has no scenario"


def _simulator(config: CueConfig, *, token: str | None) -> Any:
    if config.simulator_backend == "hf":
        return HuggingFaceSimulator(config.simulator_model or "", token=token)
    if config.simulator_backend == "openai":
        key = os.environ.get(config.api_key_env) if config.api_key_env else None
        return OpenAICompatSimulator(
            config.simulator_model or "",
            api_key=key,
            base_url=config.base_url,
        )
    raise ValueError(_INVALID_BACKEND)


def run(config: CueConfig, *, token: str | None = None) -> dict[str, Any]:
    """Execute one configured run and write stable application artifacts."""

    cue = Cue.from_pretrained(
        config.model,
        device=config.device,
        token=token,
        example_pool=config.example_pool,
        session_preprocess=config.session_preprocess,
    )
    if config.mode == "conditioned":
        if config.conversation_path is None:
            raise RuntimeError(_INVALID_CONDITIONED)
        manuals = [
            cue.from_conversation(
                load_conversation(str(config.conversation_path)),
                session_preprocess=config.session_preprocess,
                example_retrieval=config.example_retrieval,
            )
        ]
    else:
        sampled = cue.sample(
            n=config.n,
            seed=config.seed,
            example_retrieval=config.example_retrieval,
        )
        manuals = sampled if isinstance(sampled, list) else [sampled]

    output_dir = config.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    manual_payload = [manual.to_dict() for manual in manuals]
    (output_dir / "manuals.json").write_text(json.dumps(manual_payload, indent=2) + "\n", encoding="utf-8")
    (output_dir / "steering_prompts.txt").write_text(
        "\n\n---\n\n".join(manual.steering_prompt for manual in manuals) + "\n",
        encoding="utf-8",
    )

    simulated_users: list[str] = []
    if config.runs_simulator:
        history = load_conversation(str(config.history_path)) if config.history_path else []
        simulator = _simulator(config, token=token)
        if config.scenario is None:
            raise RuntimeError(_INVALID_SIMULATOR)
        simulated_users = [
            run_user_turn(
                simulator,
                scenario=config.scenario,
                history=history,
                manual=manual,
            )
            for manual in manuals
        ]
        (output_dir / "simulated_users.json").write_text(json.dumps(simulated_users, indent=2) + "\n", encoding="utf-8")

    info = {
        "config": config.public_dict(),
        "manual_count": len(manuals),
        "simulated_users": simulated_users,
        "outputs": {
            "manuals": str(output_dir / "manuals.json"),
            "steering_prompts": str(output_dir / "steering_prompts.txt"),
            "simulated_users": (str(output_dir / "simulated_users.json") if config.runs_simulator else None),
        },
    }
    (output_dir / "info.json").write_text(json.dumps(info, indent=2) + "\n", encoding="utf-8")
    return info
