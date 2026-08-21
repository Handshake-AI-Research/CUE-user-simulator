"""Resolve CUE checkpoint layouts (native cue-hf, joint training, decoder refinement)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import torch

from cue_hf.configuration_cue import CueConfig

JOINT_CONFIG_NAME = "unified_config.json"
JOINT_WEIGHTS_NAME = "model.pt"
REFINED_DECODER_STATE = "decoder_state.pt"
REFINED_CONFIG_NAME = "refinement_config.json"
JOINT_CHECKPOINT_KEY = "joint_training_checkpoint"

LAYOUT_NATIVE = "native"
LAYOUT_JOINT = "joint"
LAYOUT_REFINED = "refined"


def resolve_local_dir(
    path_or_repo: str | Path,
    *,
    revision: str | None = None,
    token: str | bool | None = None,
) -> Path:
    """Return a local directory for a path or a Hub repo id."""

    candidate = Path(path_or_repo)
    if candidate.is_dir():
        return candidate
    from huggingface_hub import snapshot_download

    return Path(snapshot_download(str(path_or_repo), revision=revision, token=token))


def detect_layout(directory: str | Path) -> str:
    directory = Path(directory)
    if _refined_config_path(directory) is not None:
        return LAYOUT_REFINED
    if (directory / JOINT_CONFIG_NAME).is_file():
        return LAYOUT_JOINT
    if (directory / "config.json").is_file():
        return LAYOUT_NATIVE
    raise ValueError(
        f"{directory} is not a cue-hf, joint training, or refinement checkpoint "
        f"(looked for config.json, {JOINT_CONFIG_NAME}, {REFINED_CONFIG_NAME})"
    )


def _refined_config_path(directory: Path) -> Path | None:
    path = directory / REFINED_CONFIG_NAME
    return path if path.is_file() else None


def config_from_joint_dir(directory: str | Path) -> CueConfig:
    payload = json.loads((Path(directory) / JOINT_CONFIG_NAME).read_text(encoding="utf-8"))
    return CueConfig.from_cue_config(payload)


def load_joint_state(directory: str | Path, *, map_location: str | torch.device = "cpu") -> dict[str, Any]:
    payload = torch.load(Path(directory) / JOINT_WEIGHTS_NAME, map_location=map_location, weights_only=False)
    return payload["model"]


def resolve_refined(directory: str | Path) -> tuple[Path, Path]:
    """Return (joint checkpoint dir, decoder state path) for a refinement checkpoint."""

    directory = Path(directory)
    config_path = _refined_config_path(directory)
    if config_path is None:
        raise ValueError(f"{directory} has no refinement config")
    payload = json.loads(config_path.read_text(encoding="utf-8"))
    joint = payload.get(JOINT_CHECKPOINT_KEY)
    if not joint:
        raise ValueError(
            f"{config_path} has no {JOINT_CHECKPOINT_KEY}; cannot find the "
            "joint training checkpoint holding the encoder"
        )
    decoder_state = directory / REFINED_DECODER_STATE
    if not decoder_state.is_file():
        raise FileNotFoundError(f"refinement checkpoint missing {decoder_state}")
    return Path(str(joint)), decoder_state
