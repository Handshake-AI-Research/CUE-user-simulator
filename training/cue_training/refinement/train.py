"""decoder refinement adversarial-probe GRPO for persona-manual decoding."""

from __future__ import annotations

import copy
import json
import os
import random
from dataclasses import replace
from datetime import timedelta
from itertools import islice
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.distributed as dist

from cue_training.data.schema import (
    COMMAND_BLOCK_SENTINEL,
    COMMAND_SLOT_SENTINEL,
    parse_manual,
)
from cue_training.infer import load_unified
from cue_training.refinement.buffer import ReplayBuffer
from cue_training.refinement.data import iter_refinement_examples, target_masked_session
from cue_training.refinement.grpo import clipped_grpo_loss
from cue_training.refinement.leakage import (
    paired_gap,
    role_compatible_future,
    save_report,
    target_manual_overlap,
)
from cue_training.refinement.probe import PairwiseOSSHiddenStateProbe
from cue_training.refinement.reward_panel import Judge, RewardPanel
from cue_training.refinement.simulator import UserSimulator
from cue_training.utils.config import get, storage_root
from cue_training.utils.log import log
from cue_training.utils.seed import seed_everything

TAG = "refinement"
REFINEMENT_CONFIG_NAME = "refinement_config.json"
JOINT_CHECKPOINT_KEY = "joint_training_checkpoint"


def _ddp_enabled() -> bool:
    return int(os.environ.get("WORLD_SIZE", "1")) > 1


def _setup_distributed(timeout_min: float = 60.0) -> tuple[int, int, int, torch.device]:
    """Return rank, local_rank, world_size, device (LOCAL_RANK under torchrun)."""

    if not _ddp_enabled():
        device_name = os.environ.get("CUE_REFINE_DEVICE") or "cuda"
        if device_name.startswith("cuda") and not torch.cuda.is_available():
            device_name = "cpu"
        return 0, 0, 1, torch.device(device_name)

    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    rank = int(os.environ.get("RANK", "0"))
    world_size = int(os.environ["WORLD_SIZE"])
    if torch.cuda.is_available():
        torch.cuda.set_device(local_rank)
        device = torch.device("cuda", local_rank)
    else:
        device = torch.device("cpu")
    if not dist.is_initialized():
        kwargs: dict[str, Any] = {
            "backend": "nccl" if device.type == "cuda" else "gloo",
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


def _average_grads(module: torch.nn.Module) -> None:
    if not (dist.is_available() and dist.is_initialized()):
        return
    for parameter in module.parameters():
        if parameter.requires_grad and parameter.grad is not None:
            dist.all_reduce(parameter.grad, op=dist.ReduceOp.AVG)


def _broadcast_object(obj: Any, *, src: int = 0) -> Any:
    if not (dist.is_available() and dist.is_initialized()):
        return obj
    payload = [obj]
    dist.broadcast_object_list(payload, src=src)
    return payload[0]


def _gather_objects(obj: Any) -> list[Any]:
    if not (dist.is_available() and dist.is_initialized()):
        return [obj]
    out: list[Any] = [None for _ in range(dist.get_world_size())]
    dist.all_gather_object(out, obj)
    return out


def refinement_config_path(checkpoint: str | Path) -> Path | None:
    """Locate a refinement config, or None when this is not a refinement checkpoint."""

    path = Path(checkpoint) / REFINEMENT_CONFIG_NAME
    return path if path.is_file() else None


def load_refinement_model(checkpoint: str | Path, *, device: str = "cpu") -> Any:
    """Load frozen joint training encoder plus a simulator-specific decoder refinement decoder."""

    checkpoint = Path(checkpoint)
    config_path = refinement_config_path(checkpoint)
    if config_path is None:
        raise FileNotFoundError(f"{checkpoint} has no {REFINEMENT_CONFIG_NAME}")
    cfg = json.loads(config_path.read_text(encoding="utf-8"))
    joint = cfg.get(JOINT_CHECKPOINT_KEY)
    if not joint:
        raise ValueError(f"{config_path} has no {JOINT_CHECKPOINT_KEY}")
    model = load_unified(joint, device=device)
    model.decoder.load_state_dict(
        torch.load(
            checkpoint / "decoder_state.pt",
            map_location=device,
            weights_only=False,
        )
    )
    # Re-tie in case the meta fallback broke lm_head<->embed_tokens (Qwen3 tied embeddings).
    if hasattr(model.decoder.lm, "tie_weights"):
        model.decoder.lm.tie_weights()
    model.eval()
    return model


def _history_text(turns: list[dict[str, Any]], max_turns: int) -> str:
    shown = turns[-max_turns:] if max_turns > 0 else turns
    return "\n".join(
        f"{str(turn.get('role') or '').upper()}: {str(turn.get('content') or '')}"
        for turn in shown
    )


def _length_penalty(candidate: str, human: str) -> float:
    gen_len, human_len = len(candidate.split()), len(human.split())
    if not gen_len or not human_len:
        return 0.0
    ratio = gen_len / human_len
    short = max((0.6 - ratio) / 0.6, 0.0)
    long = max((ratio - 1.4) / 1.4, 0.0)
    return min(0.4 * short + 0.2 * long, 0.4)


def _cue_turn_session(history: list[dict[str, Any]], turn: str) -> list[dict[str, str]]:
    """Minimal context for encoding one target turn without full-history identity leakage."""

    context = []
    for item in reversed(history):
        if item.get("role") in {"assistant", "system"}:
            context = [{"role": str(item["role"]), "content": str(item.get("content") or "")}]
            break
    return [*context, {"role": "user", "content": turn}]


@torch.no_grad()
def _encode_cue_sessions(
    model: Any,
    sessions: list[list[dict[str, str]]],
    *,
    batch_size: int,
) -> torch.Tensor:
    return torch.cat(
        [
            model.encode(sessions[start : start + max(1, batch_size)])
            for start in range(0, len(sessions), max(1, batch_size))
        ],
        dim=0,
    )


def _cue_contrastive_rewards(
    candidate_cues: torch.Tensor,
    human_cues: torch.Tensor,
    owners: torch.Tensor,
    *,
    margin: float,
    separation_weight: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Reward same-session cosine and penalize the hardest other-session match."""

    candidate_cues = torch.nn.functional.normalize(candidate_cues.float(), dim=-1)
    human_cues = torch.nn.functional.normalize(human_cues.float(), dim=-1)
    similarities = candidate_cues @ human_cues.T
    positive = similarities.gather(1, owners[:, None]).squeeze(1)
    if human_cues.size(0) < 2:
        separation = torch.zeros_like(positive)
    else:
        other = similarities.masked_fill(
            torch.nn.functional.one_hot(owners, human_cues.size(0)).bool(),
            float("-inf"),
        )
        separation = torch.relu(float(margin) + other.max(dim=1).values - positive)
    return positive - float(separation_weight) * separation, positive, separation


def _slug(value: str) -> str:
    return "".join(char if char.isalnum() else "-" for char in value).strip("-").lower()


def _sentinel_ids(model: Any, device: torch.device) -> torch.Tensor:
    sentinel = (
        COMMAND_SLOT_SENTINEL
        if bool(getattr(model, "command_slot_decode", False))
        else COMMAND_BLOCK_SENTINEL
    )
    return model.decoder.tokenizer(
        sentinel, add_special_tokens=False, return_tensors="pt"
    )["input_ids"][0].to(device)


def _make_simulator(cfg: dict[str, Any]) -> UserSimulator:
    return UserSimulator(
        model=str(get(cfg, "sim_model")),
        api_base=get(cfg, "sim_api_base"),
        api_key_env=get(cfg, "sim_api_key_env"),
        temperature=float(get(cfg, "sim_temperature", 0.7)),
        max_tokens=int(get(cfg, "sim_max_tokens", 256)),
        max_input_tokens=get(cfg, "sim_max_input_tokens"),
        system_instruction=str(get(cfg, "task_instructions", "")),
    )


def _make_panel(cfg: dict[str, Any], seed: int) -> RewardPanel | None:
    judges = []
    for raw in get(cfg, "judge_panel", []):
        key_env = raw.get("api_key_env")
        if key_env and not os.environ.get(key_env):
            continue
        judges.append(
            Judge(
                model=str(raw["model"]),
                api_base=raw.get("api_base"),
                api_key_env=key_env,
            )
        )
    if not judges:
        return None
    return RewardPanel(judges, audit_interval=1, seed=seed)


def _init_wandb(cfg: dict[str, Any]) -> Any | None:
    if bool(get(cfg, "no_wandb", False)):
        return None
    try:
        import wandb

        return wandb.init(
            entity=get(cfg, "wandb_entity") or os.environ.get("WANDB_ENTITY") or None,
            project=get(cfg, "wandb_project", "cue-unified"),
            name=get(cfg, "wandb_name", "decoder-refinement-adversarial-grpo"),
            config=cfg,
        )
    except Exception as exc:  # noqa: BLE001
        log(TAG, f"WandB disabled: {exc}")
        return None


def _wandb_log(run: Any | None, values: dict[str, Any], step: int) -> None:
    if run is not None:
        run.log(values, step=step)


@torch.no_grad()
def _generate_manuals(
    model: Any,
    sessions: list[list[dict[str, Any]]],
    *,
    sentinel_ids: torch.Tensor,
    max_new_tokens: int,
    temperature: float,
    num_candidates: int = 1,
) -> tuple[
    list[str],
    list[dict[str, Any] | None],
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    list[tuple[int, int, int]] | None,
]:
    bottleneck = model.encode(sessions)
    if bool(getattr(model, "command_slot_decode", False)):
        return model.sample_command_slot_manuals(
            bottleneck,
            num_candidates=num_candidates,
            max_new_tokens=min(max_new_tokens, 128),
            temperature=temperature,
        )
    texts, ids, mask = model.decoder.sample_manuals(
        bottleneck=bottleneck,
        sentinel_ids=sentinel_ids,
        num_candidates=num_candidates,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
    )
    return texts, [parse_manual(text) for text in texts], bottleneck, ids, mask, None


def _rollout_requests(
    examples: list[dict[str, Any]],
    manuals: list[dict[str, Any] | None],
    *,
    num_candidates: int,
) -> tuple[list[dict[str, Any]], list[int]]:
    requests: list[dict[str, Any]] = []
    indices: list[int] = []
    for index, manual in enumerate(manuals):
        if manual is None:
            continue
        example = examples[index // num_candidates]
        requests.append(
            {
                "history": example["history"],
                "task_text": example["task_text"],
                "manual": manual,
            }
        )
        indices.append(index)
    return requests, indices


def _probe_auc(
    probe: PairwiseOSSHiddenStateProbe,
    pairs: list[tuple[str, str, str]],
) -> float:
    if not pairs:
        return float("nan")
    prompts, labels = probe._training_rows(pairs, None)  # same balanced A/B protocol as fit
    features = probe._embed(prompts)
    probabilities = 1.0 / (
        1.0
        + np.exp(
            -np.clip(
                ((features - probe._mean) / probe._std) @ probe._w + probe._b,
                -500,
                500,
            )
        )
    )
    from cue_training.refinement.probe import binary_auc

    return binary_auc(labels, probabilities)


@torch.no_grad()
def _validation_pairs(
    model: Any,
    examples: list[dict[str, Any]],
    simulator: UserSimulator,
    *,
    sentinel_ids: torch.Tensor,
    max_new_tokens: int,
    max_context_turns: int,
    sim_workers: int,
    batch_size: int,
) -> tuple[list[tuple[str, str, str]], list[dict[str, Any]]]:
    pairs: list[tuple[str, str, str]] = []
    rows: list[dict[str, Any]] = []
    pending_examples: list[dict[str, Any]] = []
    pending_texts: list[str] = []
    requests: list[dict[str, Any]] = []
    for start in range(0, len(examples), max(1, batch_size)):
        chunk = examples[start : start + max(1, batch_size)]
        texts, manuals, _, _, _, _ = _generate_manuals(
            model,
            [example["full_session"] for example in chunk],
            sentinel_ids=sentinel_ids,
            max_new_tokens=max_new_tokens,
            temperature=0.0,
        )
        for example, text, manual in zip(chunk, texts, manuals, strict=True):
            if manual is None:
                continue
            pending_examples.append(example)
            pending_texts.append(text)
            requests.append(
                {
                    "history": example["history"],
                    "task_text": example["task_text"],
                    "manual": manual,
                }
            )
    turns = simulator.simulate_batch(requests, max_workers=sim_workers)
    for example, text, turn in zip(
        pending_examples, pending_texts, turns, strict=True
    ):
        if not turn:
            continue
        history = _history_text(example["history"], max_context_turns)
        pairs.append((history, example["human_turn"], turn))
        rows.append({**example, "manual_text": text, "simulator_turn": turn})
    return pairs, rows


def _run_leakage_audit(
    model: Any,
    examples: list[dict[str, Any]],
    simulator: UserSimulator,
    probe: PairwiseOSSHiddenStateProbe,
    *,
    sentinel_ids: torch.Tensor,
    epoch: int,
    output: Path,
    max_new_tokens: int,
    max_context_turns: int,
    reward_gap: float,
    overlap_threshold: float,
    seed: int,
) -> dict[str, Any]:
    deterministic_sim = replace(simulator, temperature=0.0)
    rewards = {name: [] for name in ("full", "prefix", "masked", "shuffled")}
    overlaps: list[float] = []
    rows = []
    for index, example in enumerate(examples):
        donor = examples[(index + 1) % len(examples)]["full_session"]
        conditions = {
            "full": example["full_session"],
            "prefix": example["history"],
            "masked": target_masked_session(
                example["full_session"], int(example["target_index"])
            ),
            "shuffled": role_compatible_future(
                example["history"], donor, seed=seed + index
            ),
        }
        condition_names = list(conditions)
        texts, manuals, bottlenecks, _, _, _ = _generate_manuals(
            model,
            list(conditions.values()),
            sentinel_ids=sentinel_ids,
            max_new_tokens=max_new_tokens,
            temperature=0.0,
        )
        valid_positions = [i for i, manual in enumerate(manuals) if manual is not None]
        generated_turns = deterministic_sim.simulate_batch(
            [
                {
                    "history": example["history"],
                    "task_text": example["task_text"],
                    "manual": manuals[i],
                }
                for i in valid_positions
            ],
            max_workers=4,
        )
        sim_turns = [""] * len(manuals)
        for position, turn in zip(valid_positions, generated_turns, strict=True):
            sim_turns[position] = turn
        history = _history_text(example["history"], max_context_turns)
        scored = probe.score_candidates(
            [
                (history, turn, example["human_turn"])
                for turn in sim_turns
            ]
        )
        row = {"session_id": example["session_id"], "conditions": {}}
        full_bottleneck = bottlenecks[0].float()
        for position, name in enumerate(condition_names):
            reward = min(0.8, scored[position].probability_candidate_human) - _length_penalty(
                sim_turns[position], example["human_turn"]
            )
            rewards[name].append(reward)
            overlap = target_manual_overlap(texts[position], example["human_turn"])
            if name == "full":
                overlaps.append(overlap["ngram_recall"])
            row["conditions"][name] = {
                "reward": reward,
                "manual": texts[position],
                "simulator_turn": sim_turns[position],
                "overlap": overlap,
                "bottleneck_norm": float(bottlenecks[position].norm().cpu()),
                "bottleneck_cosine_distance_from_full": float(
                    1.0
                    - torch.nn.functional.cosine_similarity(
                        full_bottleneck.unsqueeze(0),
                        bottlenecks[position].float().unsqueeze(0),
                    )
                    .cpu()
                    .item()
                ),
            }
        rows.append(row)
    gaps = {
        name: paired_gap(rewards["full"], rewards[name], seed=seed + epoch)
        for name in ("prefix", "masked", "shuffled")
    }
    mean_overlap = float(np.mean(overlaps)) if overlaps else 0.0
    detected = (
        gaps["prefix"]["ci_low"] > reward_gap
        or gaps["masked"]["ci_low"] > reward_gap
        or mean_overlap > overlap_threshold
    )
    report = {
        "epoch": epoch,
        "n": len(rows),
        "reward_means": {key: float(np.mean(value)) for key, value in rewards.items()},
        "gaps": gaps,
        "mean_target_ngram_overlap": mean_overlap,
        "leakage_detected": bool(detected),
        "rows": rows,
    }
    save_report(output, epoch, report)
    return report


def _save_state(
    output: Path,
    model: Any,
    probe: PairwiseOSSHiddenStateProbe,
    optimizer: torch.optim.Optimizer,
    buffer: ReplayBuffer,
    cfg: dict[str, Any],
    *,
    epoch: int,
    step: int,
    chance_streak: int,
    metrics: dict[str, Any],
    local_rng_state: object,
) -> None:
    output.mkdir(parents=True, exist_ok=True)
    model.decoder.save_pretrained(output / "decoder")
    torch.save(model.decoder.state_dict(), output / "decoder_state.pt")
    probe.save(output / "probe.npz")
    torch.save(optimizer.state_dict(), output / "optimizer.pt")
    buffer.save(output / "replay_buffer.json")
    torch.save(
        {
            "epoch": epoch,
            "step": step,
            "chance_streak": chance_streak,
            "python_rng": random.getstate(),
            "local_rng": local_rng_state,
            "torch_rng": torch.get_rng_state(),
            "cuda_rng": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
        },
        output / "train_state.pt",
    )
    (output / "refinement_config.json").write_text(
        json.dumps({**cfg, "last_metrics": metrics}, indent=2), encoding="utf-8"
    )


def train_refinement_from_config(cfg: dict[str, Any]) -> Path:
    rank, local_rank, world_size, device = _setup_distributed(
        timeout_min=float(get(cfg, "dist_timeout_min", 60))
    )
    is_main = rank == 0
    seed = int(get(cfg, "seed", 0)) + rank
    seed_everything(seed)
    rng = random.Random(seed)
    if not _ddp_enabled():
        # Honor explicit single-process device from config when not under torchrun.
        configured = str(get(cfg, "device", "") or "").strip()
        if configured:
            device = torch.device(configured if torch.cuda.is_available() or not configured.startswith("cuda") else "cpu")
    joint_training_checkpoint = get(cfg, "joint_training_checkpoint")
    if not joint_training_checkpoint:
        raise ValueError("refine config requires joint_training_checkpoint")
    sim_model = str(get(cfg, "sim_model"))
    output = Path(get(cfg, "output", storage_root() / "models" / "cue")) / "refinement_simulator" / _slug(
        sim_model
    )
    if is_main:
        output.mkdir(parents=True, exist_ok=True)
    if world_size > 1:
        dist.barrier()

    model = load_unified(joint_training_checkpoint, device=str(device))
    for parameter in model.encoder.parameters():
        parameter.requires_grad_(False)
    model.encoder.eval()
    model.decoder.train()
    train_decoder_lm = bool(get(cfg, "train_decoder_lm", True))
    for parameter in model.decoder.lm.parameters():
        parameter.requires_grad_(train_decoder_lm)
    resume = bool(get(cfg, "resume", True))
    if resume and (output / "decoder_state.pt").exists():
        model.decoder.load_state_dict(
            torch.load(output / "decoder_state.pt", map_location=device, weights_only=False)
        )
    reference_model = load_unified(joint_training_checkpoint, device=str(device))
    reference_decoder = reference_model.decoder.eval()
    for parameter in reference_decoder.parameters():
        parameter.requires_grad_(False)
    del reference_model  # Keep only the reference decoder; its duplicate encoder is unused.
    sentinel_ids = _sentinel_ids(model, device)

    max_train_examples = int(get(cfg, "max_train_examples", 10_000))
    train_examples = list(
        islice(
            iter_refinement_examples(
                get(cfg, "data_root"),
                split=get(cfg, "split", "train"),
                seed=int(get(cfg, "seed", 0)),
                per_source_cap=int(get(cfg, "per_source_cap", 10_000)),
            ),
            max_train_examples,
        )
    )
    validation_examples = list(
        islice(
            iter_refinement_examples(
                get(cfg, "data_root"),
                split=get(cfg, "validation_split", "validation"),
                seed=int(get(cfg, "seed", 0)) + 1,
                per_source_cap=int(get(cfg, "validation_per_source_cap", 64)),
            ),
            int(get(cfg, "probe_validation_pairs", 256)),
        )
    )
    if not train_examples or not validation_examples:
        raise ValueError("decoder refinement needs non-empty train and validation examples")

    simulator = _make_simulator(cfg)
    # Cap per-rank fan-out so N ranks don't stampede the simulator endpoint.
    sim_workers = max(1, int(get(cfg, "sim_concurrency", 32)) // world_size)
    max_context_turns = int(get(cfg, "max_context_turns", 16))
    probe_device = str(device) if world_size > 1 else str(get(cfg, "probe_device", device))
    seed_n = min(len(train_examples), int(get(cfg, "probe_seed_examples", 512)))
    # Shard seed examples across ranks for parallel probe warm-start rollouts.
    seed_examples = [
        ex
        for i, ex in enumerate(train_examples[:seed_n])
        if i % world_size == rank
    ]
    base_requests = [
        {"history": ex["history"], "task_text": ex["task_text"], "manual": None}
        for ex in seed_examples
    ]
    if is_main:
        log(
            TAG,
            f"seeding probe from {seed_n} base simulator rollouts "
            f"(~{len(seed_examples)}/rank, world_size={world_size})",
        )
    base_turns = simulator.simulate_batch(base_requests, max_workers=sim_workers) if base_requests else []
    local_seed_pairs = [
        (_history_text(ex["history"], max_context_turns), ex["human_turn"], sim)
        for ex, sim in zip(seed_examples, base_turns, strict=True)
        if sim
    ]
    seed_pairs = [
        pair
        for shard in _gather_objects(local_seed_pairs)
        for pair in shard
    ]
    probe = PairwiseOSSHiddenStateProbe(
        model_name=get(cfg, "probe_model", "Qwen/Qwen3-4B"),
        device=probe_device,
        dtype=get(cfg, "probe_dtype", "bfloat16"),
        batch_size=int(get(cfg, "probe_batch_size", 16)),
        max_length=int(get(cfg, "probe_max_length", 1024)),
        seed=int(get(cfg, "seed", 0)),
    )
    if resume and (output / "probe.npz").exists():
        if is_main:
            probe.load(output / "probe.npz")
        if world_size > 1:
            dist.barrier()
            if not is_main:
                probe.load(output / "probe.npz")
    else:
        if is_main:
            probe.fit(seed_pairs, diagnostics=True)
            probe.save(output / "probe.npz")
        if world_size > 1:
            dist.barrier()
            if not is_main:
                probe.load(output / "probe.npz")

    base_validation_pairs: list[tuple[str, str, str]] = []
    if is_main:
        base_validation_requests = [
            {"history": ex["history"], "task_text": ex["task_text"], "manual": None}
            for ex in validation_examples
        ]
        base_validation_turns = simulator.simulate_batch(
            base_validation_requests, max_workers=sim_workers
        )
        base_validation_pairs = [
            (_history_text(ex["history"], max_context_turns), ex["human_turn"], sim)
            for ex, sim in zip(validation_examples, base_validation_turns, strict=True)
            if sim
        ]
    if world_size > 1:
        base_validation_pairs = _broadcast_object(base_validation_pairs, src=0)

    optimizer = torch.optim.AdamW(
        [parameter for parameter in model.decoder.parameters() if parameter.requires_grad],
        lr=float(get(cfg, "lr", 1e-6)),
    )
    if resume and (output / "optimizer.pt").exists():
        optimizer.load_state_dict(
            torch.load(output / "optimizer.pt", map_location=device, weights_only=False)
        )
    replay_path = output / "replay_buffer.json"
    replay = (
        ReplayBuffer.load(replay_path)
        if resume and replay_path.exists() and is_main
        else ReplayBuffer(int(get(cfg, "buffer_max", 10_000)), seed=seed)
    )
    # Non-main ranks keep a local replay buffer; rank 0 owns the persisted one.
    if not is_main:
        replay = ReplayBuffer(int(get(cfg, "buffer_max", 10_000)), seed=seed)
    panel = _make_panel(cfg, seed) if is_main else None
    wandb_run = _init_wandb(cfg) if is_main else None
    num_candidates = int(get(cfg, "num_candidates", 4))
    max_new_tokens = int(get(cfg, "gen_max_new_tokens", 512))
    reward_cap = float(get(cfg, "reward_cap_p", 0.8))
    cue_reward_weight = float(get(cfg, "cue_reward_weight", 0.0))
    cue_reward_margin = float(get(cfg, "cue_reward_margin", 0.1))
    cue_reward_separation_weight = float(get(cfg, "cue_reward_separation_weight", 1.0))
    cue_reward_batch_size = int(get(cfg, "cue_reward_batch_size", 8))
    batch_sessions = int(get(cfg, "batch_sessions", 1))
    # Validation draws one candidate per session, so this many sessions puts the same
    # number of sequences in flight as a training step does.
    val_batch_sessions = int(
        get(cfg, "val_batch_sessions", batch_sessions * num_candidates)
    )
    global_step = 0
    chance_streak = 0
    start_epoch = 0
    state_path = output / "train_state.pt"
    if resume and state_path.exists():
        state = torch.load(state_path, map_location="cpu", weights_only=False)
        start_epoch = int(state.get("epoch", -1)) + 1
        global_step = int(state.get("step", 0))
        chance_streak = int(state.get("chance_streak", 0))
        # Only restore RNG on rank 0; other ranks keep rank-offset seeds for diversity.
        if is_main:
            if state.get("python_rng") is not None:
                random.setstate(state["python_rng"])
            if state.get("local_rng") is not None:
                rng.setstate(state["local_rng"])
            if state.get("torch_rng") is not None:
                torch.set_rng_state(state["torch_rng"])
            if torch.cuda.is_available() and state.get("cuda_rng") is not None:
                # A resume may run on fewer GPUs than the save did (gpus/nproc changed);
                # set_rng_state_all indexes one generator per saved state, so trim first.
                torch.cuda.set_rng_state_all(state["cuda_rng"][: torch.cuda.device_count()])

    if is_main:
        log(
            TAG,
            f"start train_examples={len(train_examples)} validation={len(validation_examples)} "
            f"N={num_candidates} simulator={sim_model} world_size={world_size} "
            f"batch_sessions/rank={batch_sessions}",
        )
    try:
        for epoch in range(start_epoch, int(get(cfg, "epochs", 10))):
            epoch_pairs: list[tuple[str, str, str]] = []
            for _ in range(int(get(cfg, "steps_per_epoch", 250))):
                examples = rng.sample(train_examples, min(batch_sessions, len(train_examples)))
                with torch.no_grad():
                    (
                        _,
                        manuals,
                        bottleneck,
                        token_ids,
                        completion_mask,
                        segments,
                    ) = _generate_manuals(
                        model,
                        [example["full_session"] for example in examples],
                        sentinel_ids=sentinel_ids,
                        max_new_tokens=max_new_tokens,
                        temperature=float(get(cfg, "sample_temperature", 0.9)),
                        num_candidates=num_candidates,
                    )
                    old_logprobs = model.decoder.manual_token_logprobs(
                        bottleneck=bottleneck,
                        sentinel_ids=sentinel_ids,
                        token_ids=token_ids,
                        completion_mask=completion_mask,
                        segments=segments,
                    )
                requests, valid_indices = _rollout_requests(
                    examples, manuals, num_candidates=num_candidates
                )
                turns = simulator.simulate_batch(requests, max_workers=sim_workers) if requests else []
                sim_turns = [""] * len(manuals)
                for index, turn in zip(valid_indices, turns, strict=True):
                    sim_turns[index] = turn
                scoring_rows = []
                scoring_indices = []
                for index, turn in enumerate(sim_turns):
                    if not turn:
                        continue
                    example = examples[index // num_candidates]
                    scoring_rows.append(
                        (
                            _history_text(example["history"], max_context_turns),
                            turn,
                            example["human_turn"],
                        )
                    )
                    scoring_indices.append(index)
                scores = probe.score_candidates(scoring_rows) if scoring_rows else []
                rewards = torch.zeros(len(manuals), device=device)
                for index, score in zip(scoring_indices, scores, strict=True):
                    example = examples[index // num_candidates]
                    turn = sim_turns[index]
                    rewards[index] = min(
                        reward_cap, score.probability_candidate_human
                    ) - _length_penalty(turn, example["human_turn"])
                    history_text = _history_text(example["history"], max_context_turns)
                    pair = (history_text, example["human_turn"], turn)
                    epoch_pairs.append(pair)
                    replay.add(
                        example["session_id"],
                        example["history"],
                        example["human_turn"],
                        turn,
                    )

                cue_metrics: dict[str, float] = {}
                if cue_reward_weight > 0 and scoring_indices:
                    human_cues = _encode_cue_sessions(
                        model,
                        [
                            _cue_turn_session(example["history"], example["human_turn"])
                            for example in examples
                        ],
                        batch_size=cue_reward_batch_size,
                    )
                    candidate_cues = _encode_cue_sessions(
                        model,
                        [
                            _cue_turn_session(
                                examples[index // num_candidates]["history"],
                                sim_turns[index],
                            )
                            for index in scoring_indices
                        ],
                        batch_size=cue_reward_batch_size,
                    )
                    owners = torch.tensor(
                        [index // num_candidates for index in scoring_indices],
                        device=candidate_cues.device,
                        dtype=torch.long,
                    )
                    cue_scores, cue_positive, cue_separation = _cue_contrastive_rewards(
                        candidate_cues,
                        human_cues,
                        owners,
                        margin=cue_reward_margin,
                        separation_weight=cue_reward_separation_weight,
                    )
                    rewards[torch.tensor(scoring_indices, device=device)] += (
                        cue_reward_weight * cue_scores.to(device)
                    )
                    cue_metrics = {
                        "cue_reward_mean": float(cue_scores.mean()),
                        "cue_positive_cosine": float(cue_positive.mean()),
                        "cue_separation_penalty": float(cue_separation.mean()),
                    }

                metrics: dict[str, float] = {}
                for _update in range(int(get(cfg, "grpo_updates_per_batch", 1))):
                    optimizer.zero_grad(set_to_none=True)
                    loss, metrics = clipped_grpo_loss(
                        model.decoder,
                        reference_decoder,
                        bottleneck=bottleneck,
                        sentinel_ids=sentinel_ids,
                        token_ids=token_ids,
                        completion_mask=completion_mask,
                        old_token_logprobs=old_logprobs,
                        rewards=rewards,
                        num_candidates=num_candidates,
                        segments=segments,
                        clip_eps=float(get(cfg, "grpo_clip_eps", 0.2)),
                        kl_beta=float(get(cfg, "kl_beta", 1e-3)),
                    )
                    loss.backward()
                    _average_grads(model.decoder)
                    torch.nn.utils.clip_grad_norm_(
                        model.decoder.parameters(), float(get(cfg, "grad_clip", 1.0))
                    )
                    optimizer.step()
                metrics.update(cue_metrics)
                global_step += 1
                metrics["parse_rate"] = sum(manual is not None for manual in manuals) / len(manuals)
                metrics["sim_empty_rate"] = sum(not turn for turn in sim_turns) / len(sim_turns)
                if is_main and global_step % 10 == 0:
                    log(
                        TAG,
                        f"epoch={epoch} step={global_step} "
                        + " ".join(f"{key}={value:.4f}" for key, value in metrics.items()),
                    )
                    _wandb_log(
                        wandb_run,
                        {f"refinement/{key}": value for key, value in metrics.items()},
                        global_step,
                    )

            gathered_pairs = [
                pair
                for shard in _gather_objects(epoch_pairs)
                for pair in shard
            ]
            if is_main:
                replay_pairs = [
                    (
                        _history_text(item.history, max_context_turns),
                        item.human_turn,
                        item.simulator_turn,
                    )
                    for item in replay.sample(int(get(cfg, "probe_refit_max_pairs", 4096)))
                ]
                probe.fit(gathered_pairs + replay_pairs, diagnostics=True)
                probe.save(output / "probe.npz")
            if world_size > 1:
                dist.barrier()
                if not is_main:
                    probe.load(output / "probe.npz")

            current_auc = 0.5
            control_auc = 0.5
            current_rows: list[dict[str, Any]] = []
            if is_main:
                current_pairs, current_rows = _validation_pairs(
                    model,
                    validation_examples,
                    simulator,
                    sentinel_ids=sentinel_ids,
                    max_new_tokens=max_new_tokens,
                    max_context_turns=max_context_turns,
                    sim_workers=sim_workers,
                    batch_size=val_batch_sessions,
                )
                current_auc = _probe_auc(probe, current_pairs)
                control_auc = _probe_auc(probe, base_validation_pairs)
            if world_size > 1:
                current_auc, control_auc = _broadcast_object(
                    (current_auc, control_auc), src=0
                )
            chance = abs(current_auc - 0.5) <= float(get(cfg, "auc_chance_band", 0.05))
            control_ok = control_auc >= float(get(cfg, "control_auc_min", 0.7))
            chance_streak = chance_streak + 1 if chance and control_ok else 0
            epoch_metrics: dict[str, Any] = {
                "probe_current_auc": current_auc,
                "probe_control_auc": control_auc,
                "probe_train_auc": (probe.train_diagnostics or {}).get("auc"),
                "chance_streak": chance_streak,
            }
            if is_main:
                log(
                    TAG,
                    f"epoch={epoch} current_auc={current_auc:.4f} control_auc={control_auc:.4f} "
                    f"chance_streak={chance_streak}",
                )
                _wandb_log(
                    wandb_run,
                    {f"refinement/{key}": value for key, value in epoch_metrics.items()},
                    global_step,
                )

            if is_main and panel is not None and epoch % int(get(cfg, "audit_every", 1)) == 0:
                audits = [
                    panel.audit(
                        step=epoch,
                        session_id=row["session_id"],
                        history=row["history"],
                        human_turn=row["human_turn"],
                        simulator_turn=row["simulator_turn"],
                        force=True,
                    )
                    for row in current_rows[:8]
                ]
                panel_scores = [
                    audit["score"] for audit in audits if audit and audit["score"] is not None
                ]
                if panel_scores:
                    epoch_metrics["panel_score"] = float(np.mean(panel_scores))
                    _wandb_log(
                        wandb_run,
                        {"refinement/panel_score": epoch_metrics["panel_score"]},
                        global_step,
                    )

            if is_main and epoch % int(get(cfg, "leakage_audit_every", 1)) == 0:
                leak_examples = validation_examples[
                    : int(get(cfg, "leakage_audit_samples", 32))
                ]
                leakage_report = _run_leakage_audit(
                    model,
                    leak_examples,
                    simulator,
                    probe,
                    sentinel_ids=sentinel_ids,
                    epoch=epoch,
                    output=output,
                    max_new_tokens=max_new_tokens,
                    max_context_turns=max_context_turns,
                    reward_gap=float(get(cfg, "leakage_reward_gap", 0.05)),
                    overlap_threshold=float(get(cfg, "leakage_overlap_threshold", 0.2)),
                    seed=int(get(cfg, "seed", 0)),
                )
                epoch_metrics["leakage_detected"] = leakage_report["leakage_detected"]
                _wandb_log(
                    wandb_run,
                    {
                        "refinement/leakage/detected": float(leakage_report["leakage_detected"]),
                        "refinement/leakage/full_minus_prefix": leakage_report["gaps"]["prefix"][
                            "mean"
                        ],
                        "refinement/leakage/full_minus_masked": leakage_report["gaps"]["masked"][
                            "mean"
                        ],
                    },
                    global_step,
                )

            if is_main:
                _save_state(
                    output,
                    model,
                    probe,
                    optimizer,
                    replay,
                    cfg,
                    epoch=epoch,
                    step=global_step,
                    chance_streak=chance_streak,
                    metrics=epoch_metrics,
                    local_rng_state=rng.getstate(),
                )
            if world_size > 1:
                dist.barrier()

            should_stop = (
                epoch + 1 >= int(get(cfg, "e_min", 2))
                and chance_streak >= int(get(cfg, "auc_patience_k", 2))
            )
            should_stop = bool(_broadcast_object(should_stop, src=0))
            if should_stop:
                if is_main:
                    log(TAG, "stopping: current-policy probe AUC is stably at chance")
                break

        if wandb_run is not None:
            wandb_run.finish()
        if is_main:
            log(TAG, f"done -> {output}")
        return output
    finally:
        _cleanup_distributed()
