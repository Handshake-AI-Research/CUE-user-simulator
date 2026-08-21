"""Encode and generate from unified checkpoints."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import torch

from cue_training.data.schema import render_manual, validate_record
from cue_training.model import UnifiedModel, build_unified_model
from cue_training.utils.checkpoint import load_config, load_model_state
from cue_training.utils.log import log

TAG = "infer"


def load_unified(checkpoint: str | Path, *, device: str = "cpu") -> UnifiedModel:
    checkpoint = Path(checkpoint)
    cfg = load_config(checkpoint)
    train_cfg = cfg.get("train") or {}
    model_cfg = cfg.get("model") or {}
    build_cfg = dict(train_cfg)
    if "encoder" in model_cfg:
        build_cfg["encoder"] = model_cfg["encoder"]
    if "decoder" in model_cfg:
        build_cfg["decoder"] = model_cfg["decoder"]
    for key in (
        "dual_decode",
        "lambda_general_ce",
        "lambda_user_specific_ce",
        "lambda_style_ce",
        "target_max_tokens",
        "command_cache_size",
        "command_slot_decode",
        "general_command_slots",
        "user_specific_command_slots",
        "style_command_slots",
        "slot_embedding_dim",
        "command_noop_weight",
        "lambda_dual_separation",
        "dual_separation_margin",
    ):
        if key in model_cfg:
            build_cfg[key] = model_cfg[key]
    # allow_meta_to_empty: if any submodule meta-inits, materialize on-device; the complete
    # checkpoint below refills real weights (avoids "Cannot copy out of meta tensor").
    model = build_unified_model(build_cfg, device=device, allow_meta_to_empty=True)
    state = load_model_state(checkpoint, map_location=device)
    incompatible = model.load_state_dict(state["model"], strict=False)
    # Qwen3 ties lm_head<->embed_tokens, so the checkpoint omits lm_head.weight. When the meta
    # fallback (to_empty) rebuilt params it broke that tie, leaving lm_head as uninitialized
    # garbage (produces gibberish manuals). Re-tie so lm_head shares the trained embeddings.
    lm = getattr(getattr(model, "decoder", None), "lm", None)
    if lm is not None and hasattr(lm, "tie_weights"):
        lm.tie_weights()
    missing = [k for k in getattr(incompatible, "missing_keys", []) if "lm_head" not in k]
    if missing:
        log(TAG, f"WARN: {len(missing)} missing checkpoint keys (e.g. {missing[:3]})")
    model.eval()
    return model


def encode_file(checkpoint: str, input_path: str, *, device: str = "cpu") -> list[float]:
    model = load_unified(checkpoint, device=device)
    record = validate_record(json.loads(Path(input_path).read_text(encoding="utf-8")))
    with torch.no_grad():
        emb = model.encode([record["turns"]])[0].detach().cpu().tolist()
    return emb


def generate_file(
    checkpoint: str, input_path: str, *, device: str = "cpu"
) -> dict[str, Any] | None:
    model = load_unified(checkpoint, device=device)
    record = validate_record(json.loads(Path(input_path).read_text(encoding="utf-8")))
    manuals = model.generate_manuals([record["turns"]])
    manual = manuals[0]
    log(TAG, f"generated={render_manual(manual)[:200]}")
    return manual
