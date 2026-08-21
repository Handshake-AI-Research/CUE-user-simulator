"""Unified checkpoint save/load."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import torch

from cue_training.utils.log import log

TAG = "ckpt"
CONFIG_NAME = "unified_config.json"
MODEL_NAME = "model.pt"
OPTIM_NAME = "optimizer.pt"
RNG_NAME = "rng.pt"


def save_checkpoint(
    output_dir: str | Path,
    *,
    config: dict[str, Any],
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer | None = None,
    scheduler: Any | None = None,
    scaler: Any | None = None,
    epoch: int = 0,
    step: int = 0,
    sampler_state: dict[str, Any] | None = None,
    extra: dict[str, Any] | None = None,
) -> Path:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "epoch": epoch,
        "step": step,
        "model": model.state_dict(),
        "sampler_state": sampler_state or {},
        "extra": extra or {},
    }
    torch.save(payload, output_dir / MODEL_NAME)
    with open(output_dir / CONFIG_NAME, "w", encoding="utf-8") as handle:
        json.dump(config, handle, indent=2, sort_keys=True)
    if optimizer is not None:
        opt_payload = {
            "optimizer": optimizer.state_dict(),
            "scheduler": None if scheduler is None else scheduler.state_dict(),
            "scaler": None if scaler is None else scaler.state_dict(),
        }
        torch.save(opt_payload, output_dir / OPTIM_NAME)
    rng = {
        "torch": torch.get_rng_state(),
        "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
        "python": os.urandom(0),  # placeholder; train loop may overwrite
    }
    try:
        import random

        rng["python"] = random.getstate()
        import numpy as np

        rng["numpy"] = np.random.get_state()
    except Exception:  # noqa: BLE001
        pass
    torch.save(rng, output_dir / RNG_NAME)
    log(TAG, f"saved checkpoint -> {output_dir}")
    return output_dir


def load_config(output_dir: str | Path) -> dict[str, Any]:
    with open(Path(output_dir) / CONFIG_NAME, encoding="utf-8") as handle:
        return json.load(handle)


def load_model_state(output_dir: str | Path, map_location: str | torch.device = "cpu") -> dict[str, Any]:
    return torch.load(Path(output_dir) / MODEL_NAME, map_location=map_location, weights_only=False)
