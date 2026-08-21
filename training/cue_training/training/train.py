"""Joint end-to-end training entrypoint (single-GPU or torchrun DDP)."""

from __future__ import annotations

import argparse
import json
import os
import random
from datetime import timedelta
from functools import partial
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader

from cue_training.augment import dropout_user_turns, perturb_session
from cue_training.data.example_pool import (
    examples_from_manual,
    eval_validation_retrieval,
)
from cue_training.data.schema import (
    COMMAND_BLOCK_SENTINEL,
    canonicalize_profile,
    filter_joint_training_target,
    parse_manual,
    split_manual_targets,
    to_profile,
    validate_manual,
)
from cue_training.data.streaming import RoundRobinStreamingDataset
from cue_training.data.wegmann_style import batch_style_overlap, load_wegmann_sidecar, sidecar_path
from cue_training.model import build_unified_model
from cue_training.utils.checkpoint import load_model_state, save_checkpoint
from cue_training.utils.config import get, load_config, storage_root
from cue_training.utils.log import log
from cue_training.utils.seed import seed_everything

try:
    import wandb
except ImportError:  # optional
    wandb = None

TAG = "train"


def _commands_only_target(canonical: str) -> str:
    manual = parse_manual(canonical) or {"commands": [], "examples": []}
    commands = manual.get("commands") or []
    if commands and isinstance(commands[0], dict):
        commands = [str(c.get("text") or "") for c in commands if str(c.get("text") or "").strip()]
    commands = [str(c).strip() for c in commands if str(c).strip()]
    if not commands:
        return json.dumps({"commands": [], "examples": []}, ensure_ascii=False, sort_keys=True)
    return canonicalize_profile({"commands": commands, "examples": []})


def _init_wandb(cfg: dict[str, Any], *, rank: int, world_size: int) -> bool:
    """Init W&B on rank 0. Returns True if logging is active."""

    if rank != 0 or bool(get(cfg, "no_wandb", False)) or wandb is None:
        return False
    if not callable(getattr(wandb, "init", None)):
        log(
            TAG,
            f"Disabling WandB: imported invalid module from {getattr(wandb, '__file__', None)}",
        )
        return False
    try:
        wandb.init(
            entity=get(cfg, "wandb_entity") or os.getenv("WANDB_ENTITY") or None,
            project=get(cfg, "wandb_project") or os.getenv("WANDB_PROJECT") or "cue-unified",
            name=get(cfg, "wandb_name") or os.getenv("WANDB_NAME") or None,
            group=get(cfg, "wandb_group") or os.getenv("WANDB_GROUP") or None,
            config={**cfg, "world_size": world_size},
        )
        return True
    except Exception as exc:
        log(TAG, f"Disabling WandB after init failed: {exc}")
        return False


def _log_wandb(
    metrics: dict[str, float],
    *,
    step: int,
    epoch: int,
    enabled: bool,
    prefix: str = "train",
) -> None:
    if not enabled or wandb is None:
        return
    payload = {f"{prefix}/{k}": v for k, v in metrics.items()}
    payload[f"{prefix}/epoch"] = float(epoch)
    wandb.log(payload, step=step)


def _finish_wandb(enabled: bool) -> None:
    if enabled and wandb is not None:
        wandb.finish()


def _wandb_run_id(enabled: bool) -> str | None:
    if enabled and wandb is not None and wandb.run is not None:
        return wandb.run.id
    return None


def _collate(
    batch: list[dict[str, Any]], *, cue_positive_turn_dropout: float = 0.0
) -> dict[str, Any]:
    sessions = [item["turns"] for item in batch]
    target_manuals = [filter_joint_training_target(item["persona_manual"]) for item in batch]
    # CE targets: {commands[str], examples[str]} (RealUserSim-style, no demographics).
    profiles = [to_profile(manual) for manual in target_manuals]
    manuals = [canonicalize_profile(profile) for profile in profiles]
    general_targets: list[str] = []
    general_commands: list[list[str]] = []
    user_specific_targets: list[str] = []
    user_specific_commands: list[list[str]] = []
    style_targets: list[str] = []
    style_commands: list[list[str]] = []
    for target_manual in target_manuals:
        general, specific, style = split_manual_targets(target_manual)
        general_targets.append(_commands_only_target(general))
        user_specific_targets.append(_commands_only_target(specific))
        style_targets.append(_commands_only_target(style))
        general_profile = parse_manual(general) or {"commands": [], "examples": []}
        specific_profile = parse_manual(specific) or {"commands": [], "examples": []}
        style_profile = parse_manual(style) or {"commands": [], "examples": []}
        general_commands.append(list(general_profile.get("commands") or []))
        user_specific_commands.append(list(specific_profile.get("commands") or []))
        style_commands.append(list(style_profile.get("commands") or []))
    return {
        "sessions": sessions,
        "targets": manuals,
        "general_targets": general_targets,
        "general_commands": general_commands,
        "user_specific_targets": user_specific_targets,
        "user_specific_commands": user_specific_commands,
        "style_targets": style_targets,
        "style_commands": style_commands,
        "commands": [profile["commands"] for profile in profiles],
        "perturbed": [
            perturb_session(dropout_user_turns(s, cue_positive_turn_dropout))
            for s in sessions
        ],
        "ids": [item["id"] for item in batch],
    }


def _build_val_loader(cfg: dict[str, Any], data_root: str) -> tuple[DataLoader | None, str]:
    """Small deterministic val loader (rank-0 only). Prefers val.jsonl, else train cap."""

    val_every = int(get(cfg, "val_every", 0))
    if val_every <= 0:
        return None, ""
    val_split = str(get(cfg, "val_split", "val"))
    cap = max(1, int(get(cfg, "val_per_source_cap", 32)))
    batch_size = max(1, int(get(cfg, "val_batch_size", get(cfg, "batch_size", 4))))
    seed = int(get(cfg, "seed", 0))
    try:
        dataset = RoundRobinStreamingDataset(
            data_root,
            split=val_split,
            per_source_cap=cap,
            shuffle_buffer=0,
            seed=seed,
            require_manual=True,
        )
        source = val_split
    except FileNotFoundError:
        train_split = str(get(cfg, "split", "train"))
        dataset = RoundRobinStreamingDataset(
            data_root,
            split=train_split,
            per_source_cap=cap,
            shuffle_buffer=0,
            seed=seed + 17,
            require_manual=True,
        )
        source = f"{train_split}:cap{cap}"
    dataset.configure(rank=0, world_size=1, epoch=0)
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        collate_fn=_collate,
        num_workers=0,
    )
    return loader, source


def _materialize_retrieval_val_records(
    data_root: str,
    cfg: dict[str, Any],
) -> list[dict[str, Any]]:
    cap = max(0, int(get(cfg, "val_retrieval_cap", 0)))
    if cap <= 0:
        return []
    val_split = str(get(cfg, "val_split", "val"))
    seed = int(get(cfg, "seed", 0))
    per_source = max(1, (cap + 7) // 8)
    try:
        dataset = RoundRobinStreamingDataset(
            data_root,
            split=val_split,
            per_source_cap=per_source,
            shuffle_buffer=0,
            seed=seed + 31,
            require_manual=True,
        )
    except FileNotFoundError:
        dataset = RoundRobinStreamingDataset(
            data_root,
            split=str(get(cfg, "split", "train")),
            per_source_cap=per_source,
            shuffle_buffer=0,
            seed=seed + 37,
            require_manual=True,
        )
    dataset.configure(rank=0, world_size=1, epoch=0)
    records: list[dict[str, Any]] = []
    for item in dataset:
        records.append(item)
        if len(records) >= cap:
            break
    return records


@torch.no_grad()
def _encode_sessions_batched(
    model: torch.nn.Module,
    records: list[dict[str, Any]],
    *,
    batch_size: int,
) -> np.ndarray:
    """Encode retrieval records in large, bounded batches on the current rank."""

    size = max(1, batch_size)
    vectors: list[np.ndarray] = []
    for start in range(0, len(records), size):
        sessions = [item["turns"] for item in records[start : start + size]]
        vectors.append(model.encode(sessions).detach().float().cpu().numpy())
    if vectors:
        return np.concatenate(vectors, axis=0)
    return np.zeros((0, int(model.encoder.bottleneck_dim)), dtype=np.float32)


@torch.no_grad()
def _run_retrieval_validation(
    model: torch.nn.Module,
    records: list[dict[str, Any]],
    *,
    seed: int,
    rank: int = 0,
    world_size: int = 1,
    batch_size: int = 256,
    gallery_frac: float = 0.8,
    k_sessions: int = 8,
    n_general: int = 2,
    n_specific: int = 2,
) -> dict[str, float]:
    if len(records) < 8:
        return {}
    rng = random.Random(seed)
    shuffled = list(records)
    rng.shuffle(shuffled)
    split_at = max(1, int(len(shuffled) * gallery_frac))
    gallery = shuffled[:split_at]
    queries = shuffled[split_at:]
    if not queries:
        queries = shuffled[-max(1, len(shuffled) // 5) :]
        gallery = shuffled[: len(shuffled) - len(queries)]
    local_gallery = gallery[rank::world_size]
    local_queries = queries[rank::world_size]
    local_gallery_vecs = _encode_sessions_batched(
        model, local_gallery, batch_size=batch_size
    )
    local_query_vecs = _encode_sessions_batched(
        model, local_queries, batch_size=batch_size
    )
    local_gallery_examples = [
        examples_from_manual(filter_joint_training_target(item["persona_manual"]))
        for item in local_gallery
    ]
    local_query_targets = []
    for item in local_queries:
        manual = filter_joint_training_target(item["persona_manual"])
        texts: list[str] = []
        for cmd in validate_manual(manual)["commands"]:
            texts.extend(str(x).strip() for x in (cmd.get("examples") or []) if str(x).strip())
        local_query_targets.append(texts)
    payload = {
        "gallery_ids": [str(item["id"]) for item in local_gallery],
        "gallery_vectors": local_gallery_vecs,
        "gallery_examples": local_gallery_examples,
        "query_ids": [str(item["id"]) for item in local_queries],
        "query_vectors": local_query_vecs,
        "query_targets": local_query_targets,
    }
    if world_size > 1:
        gathered: list[Any] = [None] * world_size
        dist.all_gather_object(gathered, payload)
    else:
        gathered = [payload]
    if rank != 0:
        return {}

    gallery_vecs = np.concatenate(
        [part["gallery_vectors"] for part in gathered], axis=0
    )
    query_vecs = np.concatenate(
        [part["query_vectors"] for part in gathered], axis=0
    )
    return eval_validation_retrieval(
        gallery_ids=[sid for part in gathered for sid in part["gallery_ids"]],
        gallery_vectors=gallery_vecs,
        gallery_examples=[
            examples for part in gathered for examples in part["gallery_examples"]
        ],
        query_ids=[sid for part in gathered for sid in part["query_ids"]],
        query_vectors=query_vecs,
        query_target_examples=[
            targets for part in gathered for targets in part["query_targets"]
        ],
        k_sessions=k_sessions,
        n_general=n_general,
        n_specific=n_specific,
    )


def _format_commands(text: str, *, max_chars: int = 1200) -> str:
    """Readable command list for logs / W&B tables."""

    manual = parse_manual(text)
    if manual is None:
        return (text or "")[:max_chars]
    cmds = manual.get("commands") or []
    if cmds and isinstance(cmds[0], str):
        lines = [f"- {c}" for c in cmds]
        examples = manual.get("examples") or []
        if examples:
            lines.append("examples:")
            lines.extend(f"  • {e}" for e in examples[:8])
        out = "\n".join(lines)
    else:
        lines = [
            f"[{c.get('kind', '')}|{c.get('form', '')}] {c.get('text', '')}".strip()
            for c in cmds
            if isinstance(c, dict)
        ]
        out = "\n".join(line for line in lines if line and line != "[]")
    return (out or text)[:max_chars]


def _materialize_val_examples(loader: DataLoader, n: int) -> list[dict[str, Any]]:
    """Fixed small set (first N) for generation compare across val steps."""

    examples: list[dict[str, Any]] = []
    if n <= 0:
        return examples
    for batch in loader:
        for i, sid in enumerate(batch["ids"]):
            examples.append(
                {
                    "id": sid,
                    "session": batch["sessions"][i],
                    "target": batch["targets"][i],
                }
            )
            if len(examples) >= n:
                return examples
    return examples


@torch.no_grad()
def _run_validation(
    model: torch.nn.Module,
    loader: DataLoader,
    *,
    max_batches: int,
    lambda_consistency: float,
    lambda_var: float,
    lambda_cov: float,
    lambda_cue_infonce: float,
    lambda_cue_style: float = 0.0,
    cue_infonce_temp: float,
    cue_overlap_target_temp: float,
    cue_hard_session: bool = False,
    style_map: dict[str, Any] | None = None,
    dual_decode: bool = False,
    lambda_general_ce: float = 0.5,
    lambda_user_specific_ce: float = 0.5,
    lambda_style_ce: float = 0.0,
    lambda_dual_separation: float | None = None,
) -> dict[str, float]:
    """Average metrics over a small val subset. Caller should eval()-mode the module."""

    totals: dict[str, float] = {}
    n = 0
    for batch_idx, batch in enumerate(loader, start=1):
        if batch_idx > max_batches:
            break
        style_targets = None
        if lambda_cue_style > 0 and style_map:
            device = next(model.parameters()).device
            style_targets = batch_style_overlap(
                [str(x) for x in batch["ids"]], style_map, device
            )
        out = model(
            sessions=batch["sessions"],
            target_texts=batch["targets"],
            lambda_consistency=lambda_consistency,
            lambda_var=lambda_var,
            lambda_cov=lambda_cov,
            lambda_cue_infonce=lambda_cue_infonce,
            lambda_cue_style=lambda_cue_style if style_targets is not None else 0.0,
            cue_infonce_temp=cue_infonce_temp,
            cue_overlap_target_temp=cue_overlap_target_temp,
            cue_hard_session=cue_hard_session,
            style_targets=style_targets,
            command_sets=(
                batch["user_specific_commands"] if dual_decode else batch["commands"]
            ),
            perturbed_sessions=batch["perturbed"],
            measure_shuffle=False,
            general_targets=batch["general_targets"],
            general_command_sets=batch["general_commands"],
            user_specific_targets=batch["user_specific_targets"],
            user_specific_command_sets=batch["user_specific_commands"],
            style_manual_targets=batch.get("style_targets"),
            style_command_sets=batch.get("style_commands"),
            dual_decode=dual_decode,
            lambda_general_ce=lambda_general_ce,
            lambda_user_specific_ce=lambda_user_specific_ce,
            lambda_style_ce=lambda_style_ce,
            lambda_dual_separation=lambda_dual_separation,
        )
        if not torch.isfinite(out["loss"]):
            continue
        metrics = dict(out["metrics"])
        metrics["loss"] = float(out["loss"].detach())
        for key, value in metrics.items():
            if isinstance(value, (int, float)) and value == value and abs(value) != float("inf"):
                totals[key] = totals.get(key, 0.0) + float(value)
        n += 1
    if n == 0:
        return {}
    return {key: value / n for key, value in totals.items()} | {"batches": float(n)}


@torch.no_grad()
def _generate_val_samples(
    model: torch.nn.Module,
    examples: list[dict[str, Any]],
    *,
    max_new_tokens: int,
    dual_decode: bool = False,
) -> list[dict[str, Any]]:
    if not examples:
        return []
    sessions = [ex["session"] for ex in examples]
    if dual_decode:
        rows = model.generate_dual_manual_texts(
            sessions, max_new_tokens=max_new_tokens, temperature=0.0
        )
        out_rows: list[dict[str, Any]] = []
        for ex, (general, specific, style, rendered) in zip(examples, rows, strict=True):
            merged = {
                "commands": (
                    list(general.get("commands") or [])
                    + list(specific.get("commands") or [])
                    + list(style.get("commands") or [])
                ),
                "examples": list(specific.get("examples") or [])
                + list(style.get("examples") or []),
            }
            pred_text = json.dumps(merged, ensure_ascii=False, sort_keys=True)
            out_rows.append(
                {
                    "id": ex["id"],
                    "target": ex["target"],
                    "target_commands": _format_commands(ex["target"]),
                    "prediction": pred_text,
                    "prediction_commands": rendered or _format_commands(pred_text),
                    "prediction_raw": rendered,
                    "parse_ok": bool(rendered.strip()),
                    "prediction_general": _format_commands(
                        json.dumps(general, ensure_ascii=False, sort_keys=True)
                    ),
                    "prediction_user_specific": _format_commands(
                        json.dumps(specific, ensure_ascii=False, sort_keys=True)
                    ),
                    "prediction_style": _format_commands(
                        json.dumps(style, ensure_ascii=False, sort_keys=True)
                    ),
                }
            )
        return out_rows
    bottleneck = model.encode(sessions)
    tok = model.decoder.tokenizer
    sentinel_ids = tok(
        COMMAND_BLOCK_SENTINEL, add_special_tokens=False, return_tensors="pt"
    )["input_ids"][0].to(bottleneck.device)
    raw_texts = model.decoder.generate(
        bottleneck=bottleneck,
        sentinel_ids=sentinel_ids,
        max_new_tokens=max_new_tokens,
    )
    rows: list[dict[str, Any]] = []
    for ex, raw in zip(examples, raw_texts, strict=True):
        manual = parse_manual(raw)
        pred_text = (
            json.dumps(manual, ensure_ascii=False, sort_keys=True) if manual is not None else raw
        )
        rows.append(
            {
                "id": ex["id"],
                "target": ex["target"],
                "target_commands": _format_commands(ex["target"]),
                "prediction": pred_text,
                "prediction_commands": _format_commands(pred_text if manual is not None else raw),
                "prediction_raw": raw,
                "parse_ok": manual is not None,
            }
        )
    return rows


def _save_val_generations(
    rows: list[dict[str, Any]],
    *,
    output: Path,
    step: int,
) -> Path:
    gen_dir = output / "val_generations"
    gen_dir.mkdir(parents=True, exist_ok=True)
    path = gen_dir / f"step_{step:06d}.jsonl"
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps({"step": step, **row}, ensure_ascii=False) + "\n")
    # Running markdown digest for quick local diff across steps.
    md_path = gen_dir / "compare.md"
    with md_path.open("a", encoding="utf-8") as handle:
        handle.write(f"\n## step {step}\n\n")
        for row in rows:
            handle.write(f"### {row['id']}\n")
            handle.write("**target**\n```\n" + row["target_commands"] + "\n```\n")
            handle.write("**prediction**\n```\n" + row["prediction_commands"] + "\n```\n\n")
    return path


def _log_val_generations_wandb(
    rows: list[dict[str, Any]],
    *,
    step: int,
    path: Path,
    enabled: bool,
) -> None:
    if not enabled or wandb is None or not rows:
        return
    table = wandb.Table(
        columns=[
            "step",
            "id",
            "parse_ok",
            "target_commands",
            "prediction_commands",
            "prediction_raw",
        ]
    )
    for row in rows:
        table.add_data(
            step,
            row["id"],
            bool(row["parse_ok"]),
            row["target_commands"],
            row["prediction_commands"],
            row["prediction_raw"][:2000],
        )
    payload: dict[str, Any] = {"val/generations": table}
    try:
        artifact = wandb.Artifact(f"val_generations_step_{step:06d}", type="val_generations")
        artifact.add_file(str(path))
        wandb.log_artifact(artifact)
    except Exception as exc:
        log(TAG, f"wandb artifact upload skipped: {exc}")
    wandb.log(payload, step=step)


def _ddp_enabled() -> bool:
    return int(os.environ.get("WORLD_SIZE", "1")) > 1


def _setup_distributed(timeout_min: float = 30.0) -> tuple[int, int, int, torch.device]:
    """Return rank, local_rank, world_size, device. No-op single-process if WORLD_SIZE<=1."""

    if not _ddp_enabled():
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        return 0, 0, 1, device

    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    rank = int(os.environ.get("RANK", "0"))
    world_size = int(os.environ["WORLD_SIZE"])
    backend = "nccl" if torch.cuda.is_available() else "gloo"
    if torch.cuda.is_available():
        torch.cuda.set_device(local_rank)
        device = torch.device("cuda", local_rank)
    else:
        device = torch.device("cpu")
    if not dist.is_initialized():
        # device_id mutes NCCL "Guessing device ID" / barrier warnings on recent torch.
        # Long timeout so ranks blocking on the barrier during rank-0 validation
        # (heavy generation) don't trip the default 10-min NCCL watchdog.
        kwargs: dict[str, Any] = {
            "backend": backend,
            "timeout": timedelta(minutes=timeout_min),
        }
        if device.type == "cuda":
            try:
                dist.init_process_group(device_id=device, **kwargs)
            except TypeError:
                dist.init_process_group(**kwargs)
        else:
            dist.init_process_group(**kwargs)
    return rank, local_rank, world_size, device


def _cleanup_distributed() -> None:
    if dist.is_available() and dist.is_initialized():
        dist.barrier()
        dist.destroy_process_group()


def _unwrap(model: torch.nn.Module) -> torch.nn.Module:
    return model.module if isinstance(model, DDP) else model


def _resolve_resume_dir(cfg: dict[str, Any], output: Path) -> Path | None:
    """Resolve joint-training resume directory. Fresh optimizer is used so LR overrides apply.

    Prefer ``last`` for continuity (most recent weights). ``best`` may lag if an early
    val win was never beaten.
    """

    raw = get(cfg, "resume", None)
    if raw in (None, False, "", "false", "False", 0):
        return None

    def _has_model(path: Path) -> bool:
        return (path / "model.pt").is_file()

    if raw in (True, "true", "True", "auto"):
        for candidate in (output / "last", output / "best", output):
            if _has_model(candidate):
                return candidate
        raise FileNotFoundError(
            f"resume={raw!r} but no model.pt under {output}/last, {output}/best, or {output}"
        )
    if raw in ("best", "last"):
        path = output / str(raw)
        if not _has_model(path):
            raise FileNotFoundError(f"resume={raw!r} missing model.pt under {path}")
        return path
    path = Path(str(raw))
    if not _has_model(path):
        raise FileNotFoundError(f"resume checkpoint missing model.pt: {path}")
    return path


def _wrap_ddp(
    model: torch.nn.Module,
    local_rank: int,
    world_size: int,
    device: torch.device,
) -> torch.nn.Module:
    if world_size <= 1:
        return model
    use_cuda = device.type == "cuda"
    return DDP(
        model,
        device_ids=[local_rank] if use_cuda else None,
        output_device=local_rank if use_cuda else None,
        find_unused_parameters=True,
    )


def train_from_config(cfg: dict[str, Any]) -> Path:
    rank, local_rank, world_size, device = _setup_distributed(
        timeout_min=float(get(cfg, "dist_timeout_min", 30.0))
    )
    is_main = rank == 0
    seed_everything(int(get(cfg, "seed", 0)) + rank)

    # Config device string is overridden by local rank under DDP.
    if world_size == 1 and get(cfg, "device"):
        device = torch.device(get(cfg, "device"))

    data_root = get(cfg, "data_root")
    if not data_root:
        raise ValueError("config requires data_root")
    output = Path(get(cfg, "output", storage_root() / "models" / "cue"))
    if is_main:
        output.mkdir(parents=True, exist_ok=True)
    if world_size > 1:
        dist.barrier()

    per_source_cap = int(get(cfg, "per_source_cap", 10_000))
    # Cap is global intent; each rank consumes a shard.
    per_rank_cap = max(1, (per_source_cap + world_size - 1) // world_size) if world_size > 1 else per_source_cap

    dataset = RoundRobinStreamingDataset(
        data_root,
        split=get(cfg, "split", "train"),
        per_source_cap=per_rank_cap,
        shuffle_buffer=int(get(cfg, "shuffle_buffer", 256)),
        seed=int(get(cfg, "seed", 0)),
        require_manual=True,
    )
    dataset.configure(rank=rank, world_size=world_size, epoch=0)
    cue_positive_turn_dropout = float(get(cfg, "cue_positive_turn_dropout", 0.0))
    if not 0.0 <= cue_positive_turn_dropout <= 1.0:
        raise ValueError("cue_positive_turn_dropout must be between 0 and 1")

    loader = DataLoader(
        dataset,
        batch_size=int(get(cfg, "batch_size", 1)),
        collate_fn=partial(
            _collate, cue_positive_turn_dropout=cue_positive_turn_dropout
        ),
        num_workers=int(get(cfg, "num_workers", 0)),
        drop_last=world_size > 1,
    )
    val_every = int(get(cfg, "val_every", 0))
    val_max_batches = max(1, int(get(cfg, "val_max_batches", 16)))
    val_gen_samples = max(0, int(get(cfg, "val_gen_samples", 4)))
    val_gen_max_tokens = max(32, int(get(cfg, "val_gen_max_tokens", 512)))
    val_loader, val_source = _build_val_loader(cfg, data_root)
    val_gen_examples: list[dict[str, Any]] = []
    # Materialize on every rank (deterministic) so generation can be sharded across GPUs.
    if val_loader is not None and val_gen_samples > 0:
        val_gen_examples = _materialize_val_examples(val_loader, val_gen_samples)
        if is_main:
            log(TAG, f"val generations: n={len(val_gen_examples)} ids={[e['id'] for e in val_gen_examples]}")

    model = build_unified_model(cfg, device=str(device))
    resume_dir = _resolve_resume_dir(cfg, output)
    start_step = 0
    start_epoch = 0
    resume_extra: dict[str, Any] = {}
    if resume_dir is not None:
        payload = load_model_state(resume_dir, map_location=str(device))
        incompatible = model.load_state_dict(payload["model"], strict=False)
        missing = [
            k for k in getattr(incompatible, "missing_keys", []) if "lm_head" not in k
        ]
        if missing and is_main:
            log(TAG, f"resume WARN: {len(missing)} missing keys (e.g. {missing[:3]})")
        start_step = int(payload.get("step", 0))
        start_epoch = int(payload.get("epoch", 0))
        resume_extra = dict(payload.get("extra") or {})
        if is_main:
            log(
                TAG,
                f"resumed weights from {resume_dir} "
                f"step={start_step} epoch={start_epoch} "
                f"(fresh optimizer; config LRs apply)",
            )

    lm_format_warmup_steps = max(0, int(get(cfg, "lm_format_warmup_steps", 200)))
    # Past format warmup when continuing a conditioning-phase checkpoint.
    if resume_dir is not None and start_step >= lm_format_warmup_steps:
        lm_format_warmup_steps = 0
    encoder_trainable = {
        name: param.requires_grad for name, param in model.encoder.named_parameters()
    }

    def _set_training_phase(raw: torch.nn.Module, *, format_warmup: bool) -> None:
        for name, param in raw.encoder.named_parameters():
            param.requires_grad_(False if format_warmup else encoder_trainable[name])
        for param in raw.decoder.lm.parameters():
            param.requires_grad_(format_warmup)
        conditioning_modules = [raw.decoder.memory, raw.decoder.gated_blocks]
        if raw.decoder.slot_embeddings is not None:
            conditioning_modules.extend(
                [raw.decoder.slot_embeddings, raw.decoder.slot_projection]
            )
        for module in conditioning_modules:
            for param in module.parameters():
                param.requires_grad_(not format_warmup)
        if format_warmup:
            raw.encoder.eval()
            raw.decoder.lm.train()
        else:
            raw.encoder.train()
            raw.decoder.lm.eval()

    def _trainable_counts(raw: torch.nn.Module) -> dict[str, int]:
        conditioning_modules = [raw.decoder.memory, raw.decoder.gated_blocks]
        if raw.decoder.slot_embeddings is not None:
            conditioning_modules.extend(
                [raw.decoder.slot_embeddings, raw.decoder.slot_projection]
            )
        return {
            "encoder": sum(p.numel() for p in raw.encoder.parameters() if p.requires_grad),
            "lm": sum(p.numel() for p in raw.decoder.lm.parameters() if p.requires_grad),
            "conditioning": sum(
                p.numel()
                for module in conditioning_modules
                for p in module.parameters()
                if p.requires_grad
            ),
        }

    in_format_warmup = lm_format_warmup_steps > 0
    _set_training_phase(model, format_warmup=in_format_warmup)
    if is_main:
        log(TAG, f"trainable={_trainable_counts(model)}")

    model = _wrap_ddp(model, local_rank, world_size, device)

    lr = float(get(cfg, "lr", 5e-5))
    lr_backbone = float(get(cfg, "lr_backbone", 5e-6))
    lr_lm = float(get(cfg, "lr_lm", 5e-6))
    lr_gate = float(get(cfg, "lr_gate", 1e-5))
    weight_decay = float(get(cfg, "weight_decay", 0.01))

    def _make_optim(m: torch.nn.Module) -> torch.optim.Optimizer:
        # e5 backbone needs ~5e-6 (cue encoder default); 1e-4 blows up after a few steps.
        # Gate scalars get dL/dα ≈ attended (large) even when tanh(α)≈0 — keep lr_gate small.
        raw = _unwrap(m)
        backbone_ids = {id(p) for p in raw.encoder.user_encoder.parameters()}
        lm_ids = {id(p) for p in raw.decoder.lm.parameters()}
        gate_params = [
            p
            for block in raw.decoder.gated_blocks
            for p in (block.alpha_xattn, block.alpha_dense)
        ]
        gate_ids = {id(p) for p in gate_params}
        groups: list[dict[str, Any]] = []
        backbone = [p for p in raw.encoder.user_encoder.parameters() if p.requires_grad]
        lm = [p for p in raw.decoder.lm.parameters() if p.requires_grad]
        gates = [p for p in gate_params if p.requires_grad]
        rest = [
            p
            for p in m.parameters()
            if p.requires_grad
            and id(p) not in backbone_ids
            and id(p) not in lm_ids
            and id(p) not in gate_ids
        ]
        if backbone:
            groups.append({"params": backbone, "lr": lr_backbone, "weight_decay": weight_decay})
        if lm:
            groups.append({"params": lm, "lr": lr_lm, "weight_decay": weight_decay})
        if rest:
            groups.append({"params": rest, "lr": lr, "weight_decay": weight_decay})
        if gates:
            groups.append({"params": gates, "lr": lr_gate, "weight_decay": 0.0})
        if not groups:
            raise RuntimeError("no trainable parameters for optimizer")
        return torch.optim.AdamW(groups)

    optim = _make_optim(model)
    epochs = int(get(cfg, "epochs", 1))
    grad_accum = max(1, int(get(cfg, "grad_accum", 1)))
    max_steps = int(get(cfg, "max_steps", 0))
    lambda_consistency = float(get(cfg, "lambda_consistency", 0.1))
    lambda_var = float(get(cfg, "lambda_var", 1.0))
    lambda_cov = float(get(cfg, "lambda_cov", 0.04))
    lambda_cue_infonce = float(get(cfg, "lambda_cue_infonce", 0.0))
    lambda_cue_style = float(get(cfg, "lambda_cue_style", 0.0))
    cue_infonce_temp = float(get(cfg, "cue_infonce_temp", 0.1))
    cue_overlap_target_temp = float(get(cfg, "cue_overlap_target_temp", 0.2))
    cue_hard_session = bool(get(cfg, "cue_hard_session", False))
    dual_decode = bool(get(cfg, "dual_decode", False))
    lambda_general_ce = float(get(cfg, "lambda_general_ce", 0.5))
    lambda_user_specific_ce = float(get(cfg, "lambda_user_specific_ce", 0.5))
    lambda_style_ce = float(get(cfg, "lambda_style_ce", 0.0))
    clip = float(get(cfg, "grad_clip", 1.0))

    style_sidecar = get(cfg, "wegmann_style_path") or sidecar_path(
        data_root, split=str(get(cfg, "split", "train"))
    )
    style_map = load_wegmann_sidecar(style_sidecar) if lambda_cue_style > 0 else {}
    if lambda_cue_style > 0 and not style_map and is_main:
        log(
            TAG,
            f"lambda_cue_style={lambda_cue_style} but no sidecar at {style_sidecar}; "
            "run `cue precompute-wegmann-style` first",
        )
    val_retrieval_batch_size = max(
        1, int(get(cfg, "val_retrieval_batch_size", 256))
    )
    # Every rank materializes the same deterministic records, then encodes its rank-strided shard.
    val_retrieval_records = _materialize_retrieval_val_records(data_root, cfg)
    if is_main and val_retrieval_records:
        log(
            TAG,
            f"val retrieval subset n={len(val_retrieval_records)} "
            f"batch_size_per_gpu={val_retrieval_batch_size}",
        )

    use_wandb = _init_wandb(cfg, rank=rank, world_size=world_size)
    log_every = max(1, int(get(cfg, "log_every", 10)))

    best_val_loss = float(resume_extra.get("best_val_loss", float("inf")))
    if best_val_loss != best_val_loss:  # NaN guard
        best_val_loss = float("inf")
    # Periodic last/ saves: default to val_every (or 100 if val is off).
    save_every = int(get(cfg, "save_every", val_every if val_every > 0 else 100))
    save_every = max(0, save_every)

    def _checkpoint_extra(*, val_loss: float | None = None) -> dict[str, Any]:
        extra: dict[str, Any] = {
            "best_val_loss": best_val_loss,
            "wandb_run_id": _wandb_run_id(use_wandb),
            "training_phase": (
                "format_warmup" if in_format_warmup else "conditioning"
            ),
        }
        if val_loss is not None:
            extra["val_loss"] = val_loss
        return extra

    def _save_last(
        step: int, epoch: int, *, val_loss: float | None = None
    ) -> None:
        if not is_main:
            return
        raw_m = _unwrap(model)
        last_dir = output / "last"
        save_checkpoint(
            last_dir,
            config={"train": cfg, "model": raw_m.config_dict()},
            model=raw_m,
            optimizer=optim,
            epoch=epoch,
            step=step,
            sampler_state=dataset.state_dict(),
            extra=_checkpoint_extra(val_loss=val_loss),
        )
        raw_m.decoder.save_pretrained(last_dir / "decoder")
        log(TAG, f"saved last checkpoint step={step} epoch={epoch} -> {last_dir}")

    def _maybe_validate(step: int, epoch: int, *, force: bool = False) -> None:
        nonlocal best_val_loss
        if val_loader is None or val_every <= 0:
            return
        if not force and step % val_every != 0:
            return
        if world_size > 1:
            dist.barrier()

        raw_m = _unwrap(model)
        raw_m.eval()
        # Generation is the expensive part; shard examples across all ranks.
        shard_rows = _generate_val_samples(
            raw_m,
            val_gen_examples[rank::world_size],
            max_new_tokens=val_gen_max_tokens,
            dual_decode=dual_decode,
        )
        if world_size > 1:
            gathered: list[Any] = [None] * world_size
            dist.all_gather_object(gathered, shard_rows)
        else:
            gathered = [shard_rows]

        if is_main:
            val_metrics = _run_validation(
                raw_m,
                val_loader,
                max_batches=val_max_batches,
                lambda_consistency=0.0 if in_format_warmup else lambda_consistency,
                lambda_var=0.0 if in_format_warmup else lambda_var,
                lambda_cov=0.0 if in_format_warmup else lambda_cov,
                lambda_cue_infonce=0.0 if in_format_warmup else lambda_cue_infonce,
                lambda_cue_style=0.0 if in_format_warmup else lambda_cue_style,
                cue_infonce_temp=cue_infonce_temp,
                cue_overlap_target_temp=cue_overlap_target_temp,
                cue_hard_session=cue_hard_session,
                style_map=style_map if not in_format_warmup else None,
                dual_decode=dual_decode,
                lambda_general_ce=lambda_general_ce,
                lambda_user_specific_ce=lambda_user_specific_ce,
                lambda_style_ce=0.0 if in_format_warmup else lambda_style_ce,
                lambda_dual_separation=(
                    0.0 if in_format_warmup else None
                ),
            )
            # Reassemble sharded generations into the original example order.
            gen_rows = [
                gathered[i % world_size][i // world_size]
                for i in range(len(val_gen_examples))
                if i // world_size < len(gathered[i % world_size])
            ]
            if val_metrics:
                log(
                    TAG,
                    f"val step={step} source={val_source} "
                    + " ".join(f"{k}={v:.4f}" for k, v in val_metrics.items()),
                )
                _log_wandb(
                    val_metrics, step=step, epoch=epoch, enabled=use_wandb, prefix="val"
                )
                val_loss = val_metrics.get("loss")
                # Always refresh last/ on val so resume can pick up recent weights.
                _save_last(step, epoch, val_loss=val_loss)
                if (
                    not in_format_warmup
                    and val_loss is not None
                    and val_loss < best_val_loss
                ):
                    best_val_loss = val_loss
                    best_dir = output / "best"
                    save_checkpoint(
                        best_dir,
                        config={"train": cfg, "model": raw_m.config_dict()},
                        model=raw_m,
                        optimizer=optim,
                        epoch=epoch,
                        step=step,
                        sampler_state=dataset.state_dict(),
                        extra=_checkpoint_extra(val_loss=val_loss),
                    )
                    raw_m.decoder.save_pretrained(best_dir / "decoder")
                    log(TAG, f"new best val_loss={val_loss:.4f} -> {best_dir}")
                elif in_format_warmup and val_loss is not None and is_main:
                    log(
                        TAG,
                        f"skip best save during format warmup (val_loss={val_loss:.4f})",
                    )
            if gen_rows:
                gen_path = _save_val_generations(gen_rows, output=output, step=step)
                for row in gen_rows[:2]:
                    log(
                        TAG,
                        f"val gen id={row['id']} parse_ok={row['parse_ok']}\n"
                        f"  target: {row['target_commands'][:240]}\n"
                        f"  pred:   {row['prediction_commands'][:240]}",
                    )
                log(TAG, f"val generations -> {gen_path}")
                _log_val_generations_wandb(
                    gen_rows, step=step, path=gen_path, enabled=use_wandb
                )
        if force and val_retrieval_records and not in_format_warmup:
            retrieval_metrics = _run_retrieval_validation(
                raw_m,
                val_retrieval_records,
                seed=int(get(cfg, "seed", 0)) + step,
                rank=rank,
                world_size=world_size,
                batch_size=val_retrieval_batch_size,
                k_sessions=int(get(cfg, "example_retrieval_k_sessions", 8)),
                n_general=int(get(cfg, "example_retrieval_n_general", 2)),
                n_specific=int(get(cfg, "example_retrieval_n_specific", 2)),
            )
            if is_main and retrieval_metrics:
                log(
                    TAG,
                    f"val retrieval step={step} "
                    + " ".join(f"{k}={v:.4f}" for k, v in retrieval_metrics.items()),
                )
                _log_wandb(
                    retrieval_metrics,
                    step=step,
                    epoch=epoch,
                    enabled=use_wandb,
                    prefix="val",
                )
        raw_m.train()
        if in_format_warmup:
            raw_m.encoder.eval()
        else:
            raw_m.decoder.lm.eval()
        if world_size > 1:
            dist.barrier()

    if is_main:
        log(
            TAG,
            f"start world_size={world_size} device={device} batch_size={get(cfg, 'batch_size', 1)} "
            f"grad_accum={grad_accum} per_rank_cap={per_rank_cap} "
            f"lr={lr} lr_backbone={lr_backbone} lr_lm={lr_lm} lr_gate={lr_gate} wandb={use_wandb} "
            f"phase={'format_warmup' if in_format_warmup else 'conditioning'} "
            f"lm_format_warmup_steps={lm_format_warmup_steps} "
            f"val_every={val_every} val_source={val_source or 'off'} "
            f"val_max_batches={val_max_batches} val_gen_samples={len(val_gen_examples)} "
            f"dual_decode={dual_decode} positive_turn_dropout={cue_positive_turn_dropout} "
            f"command_slots={get(cfg, 'command_slot_decode', False)} "
            f"slot_counts={get(cfg, 'general_command_slots', 5)}+"
            f"{get(cfg, 'user_specific_command_slots', 5)} "
            f"lambda_cue_style={lambda_cue_style} style_vectors={len(style_map)} "
            f"resume={resume_dir or 'none'} "
            f"start_step={start_step} start_epoch={start_epoch} save_every={save_every}",
        )

    step = start_step
    skipped_nonfinite = 0
    model.train()
    if in_format_warmup:
        _unwrap(model).encoder.eval()
    else:
        _unwrap(model).decoder.lm.eval()
    for epoch in range(start_epoch, epochs):
        dataset.set_epoch(epoch)
        optim.zero_grad(set_to_none=True)
        loader_iter = iter(loader)
        batch_idx = 0
        while True:
            batch = next(loader_iter, None)
            # End the epoch on all ranks together as soon as any shard is exhausted,
            # so per-rank step counters never drift (uneven shards otherwise desync
            # the validation barrier / DDP gradient all-reduce at epoch boundaries).
            if world_size > 1:
                have_batch = torch.tensor(
                    [1.0 if batch is not None else 0.0], device=device
                )
                dist.all_reduce(have_batch, op=dist.ReduceOp.MIN)
                if have_batch.item() < 1.0:
                    break
            elif batch is None:
                break
            batch_idx += 1
            if in_format_warmup and step == lm_format_warmup_steps:
                if is_main:
                    log(TAG, "format warm-up complete; freezing LM and enabling CUE conditioning")
                raw = _unwrap(model)
                _set_training_phase(raw, format_warmup=False)
                if is_main:
                    log(TAG, f"trainable={_trainable_counts(raw)}")
                model = _wrap_ddp(raw, local_rank, world_size, device)
                optim = _make_optim(model)
                in_format_warmup = False

            out = model(
                sessions=batch["sessions"],
                target_texts=batch["targets"],
                lambda_consistency=0.0 if in_format_warmup else lambda_consistency,
                lambda_var=0.0 if in_format_warmup else lambda_var,
                lambda_cov=0.0 if in_format_warmup else lambda_cov,
                lambda_cue_infonce=0.0 if in_format_warmup else lambda_cue_infonce,
                lambda_cue_style=0.0 if in_format_warmup else lambda_cue_style,
                cue_infonce_temp=cue_infonce_temp,
                cue_overlap_target_temp=cue_overlap_target_temp,
                cue_hard_session=cue_hard_session,
                style_targets=(
                    batch_style_overlap(
                        [str(x) for x in batch["ids"]], style_map, device
                    )
                    if (not in_format_warmup and lambda_cue_style > 0 and style_map)
                    else None
                ),
                command_sets=(
                batch["user_specific_commands"] if dual_decode else batch["commands"]
            ),
                perturbed_sessions=None if in_format_warmup else batch["perturbed"],
                measure_shuffle=(step % 50 == 0),
                general_targets=batch["general_targets"],
                general_command_sets=batch["general_commands"],
                user_specific_targets=batch["user_specific_targets"],
                user_specific_command_sets=batch["user_specific_commands"],
                style_manual_targets=batch["style_targets"],
                style_command_sets=batch["style_commands"],
                dual_decode=dual_decode,
                lambda_general_ce=lambda_general_ce,
                lambda_user_specific_ce=lambda_user_specific_ce,
                lambda_style_ce=0.0 if in_format_warmup else lambda_style_ce,
                lambda_dual_separation=0.0 if in_format_warmup else None,
            )
            # All ranks must agree (DDP); skip if any rank has a non-finite loss.
            finite = torch.tensor(
                [1.0 if torch.isfinite(out["loss"]) else 0.0], device=device
            )
            if world_size > 1:
                dist.all_reduce(finite, op=dist.ReduceOp.MIN)
            if finite.item() < 1.0:
                skipped_nonfinite += 1
                if is_main and skipped_nonfinite <= 5:
                    parts = " ".join(
                        f"{k}={'ok' if isinstance(v, float) and v == v and abs(v) != float('inf') else repr(v)}"
                        for k, v in out["metrics"].items()
                    )
                    log(
                        TAG,
                        f"non-finite loss; skipping batch (count={skipped_nonfinite}) {parts}",
                    )
                optim.zero_grad(set_to_none=True)
                continue

            (out["loss"] / grad_accum).backward()
            if batch_idx % grad_accum == 0:
                if clip > 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
                optim.step()
                optim.zero_grad(set_to_none=True)
                step += 1
                if is_main and step % log_every == 0:
                    metrics = dict(out["metrics"])
                    metrics["loss"] = float(out["loss"].detach())
                    log(
                        TAG,
                        f"epoch={epoch} step={step} "
                        + " ".join(f"{k}={v:.4f}" for k, v in metrics.items()),
                    )
                    _log_wandb(metrics, step=step, epoch=epoch, enabled=use_wandb)
                _maybe_validate(step, epoch)
                # Extra last/ saves when save_every is finer than val (or val is off).
                if (
                    save_every > 0
                    and step % save_every == 0
                    and (val_every <= 0 or step % val_every != 0)
                ):
                    if world_size > 1:
                        dist.barrier()
                    _save_last(step, epoch)
                    if world_size > 1:
                        dist.barrier()
                if max_steps and step >= max_steps:
                    break
        if world_size > 1:
            dist.barrier()
        _save_last(step, epoch)
        if world_size > 1:
            dist.barrier()
        if max_steps and step >= max_steps:
            break

    _maybe_validate(step, epochs - 1, force=True)

    if world_size > 1:
        dist.barrier()

    raw = _unwrap(model)
    if is_main:
        _save_last(step, epochs)
        save_checkpoint(
            output,
            config={"train": cfg, "model": raw.config_dict()},
            model=raw,
            optimizer=optim,
            epoch=epochs,
            step=step,
            sampler_state=dataset.state_dict(),
            extra=_checkpoint_extra(),
        )
        raw.decoder.save_pretrained(output / "decoder")
        log(TAG, f"done -> {output}")
        _finish_wandb(use_wandb)

    if world_size > 1:
        dist.barrier()
    _cleanup_distributed()
    return output


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train cue jointly.")
    p.add_argument("--config", required=True)
    p.add_argument("--data_root", default=None)
    p.add_argument("--output", default=None)
    p.add_argument(
        "--resume",
        default=None,
        help="Checkpoint dir, or auto/last/best (auto prefers last). Fresh optimizer; config LRs apply.",
    )
    p.add_argument("--wandb_entity", default=None)
    p.add_argument("--wandb_project", default=None)
    p.add_argument("--wandb_name", default=None)
    p.add_argument("--wandb_group", default=None)
    p.add_argument("--no_wandb", action="store_true")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)
    if args.data_root:
        cfg["data_root"] = args.data_root
    if args.output:
        cfg["output"] = args.output
    if args.resume is not None:
        cfg["resume"] = args.resume
    if args.wandb_entity is not None:
        cfg["wandb_entity"] = args.wandb_entity
    if args.wandb_project is not None:
        cfg["wandb_project"] = args.wandb_project
    if args.wandb_name is not None:
        cfg["wandb_name"] = args.wandb_name
    if args.wandb_group is not None:
        cfg["wandb_group"] = args.wandb_group
    if args.no_wandb:
        cfg["no_wandb"] = True
    train_from_config(cfg)


if __name__ == "__main__":
    main()
