"""Train the cue embedding diffusion sampler."""

from __future__ import annotations

import json
import os
from collections import Counter
from datetime import timedelta
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.distributed as dist
import torch.nn.functional as F
try:
    from torch.amp import GradScaler as _AmpGradScaler
    from torch.amp import autocast as _amp_autocast

    def _make_scaler(enabled: bool) -> Any:
        return _AmpGradScaler("cuda", enabled=enabled)

    def _autocast(enabled: bool):
        return _amp_autocast("cuda", enabled=enabled)

except ImportError:  # pragma: no cover
    from torch.cuda.amp import GradScaler as _AmpGradScaler  # type: ignore
    from torch.cuda.amp import autocast as _amp_autocast  # type: ignore

    def _make_scaler(enabled: bool) -> Any:
        return _AmpGradScaler(enabled=enabled)

    def _autocast(enabled: bool):
        return _amp_autocast(enabled=enabled)

from cue_training.sampler.bank import EmbeddingBank, load_bank
from cue_training.utils.config import storage_root
from cue_training.sampler.diffusion import CosineNoiseSchedule, DiffusionSamplerModel, EMA
from cue_training.sampler.metrics import (
    acceptance_report,
    decode_parse_rate,
    denoising_mse,
    evaluate_samples_vs_bank,
)
from cue_training.sampler.sample import ddim_sample, project_layernorm_manifold
from cue_training.utils.log import log, warn
from cue_training.utils.seed import seed_everything

TAG = "sampler.train"


def _setup_distributed(timeout_min: float = 60.0) -> tuple[int, int, int, str]:
    """Return rank, local_rank, world_size, device. No-op single-process if WORLD_SIZE<=1."""

    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    if world_size <= 1:
        return 0, 0, 1, ""
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    rank = int(os.environ.get("RANK", "0"))
    if torch.cuda.is_available():
        torch.cuda.set_device(local_rank)
        device = f"cuda:{local_rank}"
        backend = "nccl"
    else:
        device = "cpu"
        backend = "gloo"
    if not dist.is_initialized():
        # Validation work is sharded, so the only one-sided wait left is rank 0 writing
        # checkpoints; the generous timeout still covers a slow shared filesystem.
        dist.init_process_group(backend=backend, timeout=timedelta(minutes=timeout_min))
    return rank, local_rank, world_size, device


def _cleanup_distributed() -> None:
    if dist.is_available() and dist.is_initialized():
        dist.barrier()
        dist.destroy_process_group()


def _unwrap(model: torch.nn.Module) -> torch.nn.Module:
    return model.module if isinstance(model, torch.nn.parallel.DistributedDataParallel) else model


def _all_reduce_sum(values: list[float], device: str) -> list[float]:
    """Sum a few scalars across ranks (nccl needs them on the rank's own GPU)."""

    tensor = torch.tensor(
        values, dtype=torch.float64, device=device if device.startswith("cuda") else "cpu"
    )
    dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
    return [float(v) for v in tensor.tolist()]


def _load_decoder_everywhere(checkpoint: str, device: str, world_size: int) -> Any | None:
    """Load the decoder on this rank, returning None unless *every* rank succeeded.

    Validation aggregates with collectives, so a rank that quietly skipped decoding after a
    failed load would hang the ranks that did not. Agreeing here keeps the skip collective.
    """

    from cue_training.infer.export import load_checkpoint

    model = None
    try:
        model = load_checkpoint(checkpoint, device=device)
    except Exception as exc:  # noqa: BLE001
        warn(TAG, f"val decode skipped, could not load {checkpoint}: {exc}")
    if world_size <= 1:
        return model
    if _all_reduce_sum([1.0 if model is not None else 0.0], device)[0] < world_size:
        return None
    return model


def _decode_parse_sharded(
    model: Any,
    samples: np.ndarray,
    *,
    max_new_tokens: int,
    batch_size: int,
    rank: int,
    world_size: int,
    device: str,
) -> dict[str, float] | None:
    """Decode a disjoint slice per rank, then aggregate to the whole-set metrics.

    Sharding is what keeps generation off the critical path of a single rank; the pooled
    counts and texts make the result identical to decoding every row in one process.
    Returns None when any rank failed, so all of them skip together.
    """

    rows = samples[rank::world_size] if world_size > 1 else samples
    local: dict[str, Any] | None = None
    try:
        local = (
            decode_parse_rate(
                model, rows, max_new_tokens=max_new_tokens, batch_size=batch_size
            )
            if len(rows)
            else {"parse_rate": 0.0, "mean_length": 0.0, "n": 0.0, "texts": []}
        )
    except Exception as exc:  # noqa: BLE001
        warn(TAG, f"val decode failed on rank {rank}: {exc}")
    if world_size <= 1:
        return local
    if _all_reduce_sum([1.0 if local is not None else 0.0], device)[0] < world_size:
        return None
    parsed, length_sum, total = _all_reduce_sum(
        [
            local["parse_rate"] * local["n"],
            local["mean_length"] * local["n"],
            local["n"],
        ],
        device,
    )
    pooled: list[list[str]] = [[] for _ in range(world_size)]
    dist.all_gather_object(pooled, local["texts"])
    texts = [text for chunk in pooled for text in chunk]
    return {
        "parse_rate": parsed / max(total, 1.0),
        "mean_length": length_sum / max(total, 1.0),
        # Pooled, so duplicates straddling two shards still count.
        "duplicate_rate": (1.0 - len(set(texts)) / len(texts)) if texts else 0.0,
        "n": total,
    }


def _source_weights(sources: list[str], alpha: float) -> np.ndarray | None:
    """Row probabilities putting mass ``n_source ** alpha`` on each source.

    ``alpha=1`` is uniform over rows (source mass tracks how many sessions each corpus
    contributed) and returns None; ``alpha=0`` gives every source equal mass. The
    annotation per-dataset cap makes the row mixture an artifact of that cap, so this is
    the knob for choosing which users the sampler models.
    """

    if alpha == 1.0 or not sources:
        return None
    counts = Counter(sources)
    weights = np.array([counts[s] ** (alpha - 1.0) for s in sources], dtype=np.float64)
    total = weights.sum()
    if total <= 0:
        return None
    return weights / total


def _sample_neighborhood_episode(
    embeddings: np.ndarray,
    index: Any,
    *,
    batch_size: int,
    k: int,
    rng: np.random.Generator,
    p_uncond: float,
    weights: np.ndarray | None = None,
    source_codes: np.ndarray | None = None,
    same_source_frac: float = 0.0,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return (z0 [B,D], set [B,K,D], mask [B,K], drop [B]) as float/bool tensors."""

    n = embeddings.shape[0]
    if n == 0:
        raise RuntimeError("empty embedding shard")
    k = max(1, min(k, n - 1 if n > 1 else 1))
    # Only the anchor is reweighted; its neighborhood stays whatever kNN returns.
    centers = (
        rng.choice(n, size=batch_size, p=weights)
        if weights is not None
        else rng.integers(0, n, size=batch_size)
    )
    restrict = source_codes is not None and same_source_frac > 0.0
    # Dropping other-source neighbors needs a deeper candidate list to still fill k slots.
    sims, nbrs = index.search(
        embeddings[centers], min(n, k * 4 + 1) if restrict else k + 1
    )
    z0_list = []
    sets = []
    masks = []
    drops = []
    for i in range(batch_size):
        ids = [int(x) for x in nbrs[i].tolist() if int(x) >= 0 and int(x) != int(centers[i])]
        if restrict:
            if rng.random() < same_source_frac:
                same = [j for j in ids if source_codes[j] == source_codes[centers[i]]]
                # Falling back to the mixed list beats dropping an episode outright.
                ids = same or ids
            ids = ids[:k]
        if not ids:
            ids = [int(centers[i])]
        # target: random member of neighborhood (or center)
        pool = ids + [int(centers[i])]
        target = int(rng.choice(pool))
        cond_pool = [j for j in ids if j != target]
        if not cond_pool:
            cond_pool = [target]
        m = int(rng.integers(1, min(k, len(cond_pool)) + 1))
        chosen = list(rng.choice(cond_pool, size=m, replace=False))
        pad = k - len(chosen)
        set_vecs = [embeddings[j] for j in chosen] + [np.zeros_like(embeddings[0])] * pad
        mask = [True] * len(chosen) + [False] * pad
        z0_list.append(embeddings[target])
        sets.append(np.stack(set_vecs, axis=0))
        masks.append(mask)
        drops.append(bool(rng.random() < p_uncond))
    z0 = torch.from_numpy(np.stack(z0_list).astype(np.float32))
    set_t = torch.from_numpy(np.stack(sets).astype(np.float32))
    mask_t = torch.tensor(masks, dtype=torch.bool)
    drop_t = torch.tensor(drops, dtype=torch.bool)
    _ = sims
    return z0, set_t, mask_t, drop_t


def _infonce(z_hat: torch.Tensor, z0: torch.Tensor, temperature: float = 0.1) -> torch.Tensor:
    a = F.normalize(z_hat, dim=-1)
    b = F.normalize(z0.detach(), dim=-1)
    logits = (a @ b.T) / max(temperature, 1e-6)
    labels = torch.arange(a.size(0), device=a.device)
    return F.cross_entropy(logits, labels)


def _save_sampler_checkpoint(
    output: Path,
    *,
    model: DiffusionSamplerModel,
    ema: EMA,
    optimizer: torch.optim.Optimizer,
    scheduler: Any,
    scaler: Any,
    schedule: CosineNoiseSchedule,
    bank: EmbeddingBank,
    cfg: dict[str, Any],
    step: int,
    best_metric: float,
    extra: dict[str, Any] | None = None,
) -> None:
    output.mkdir(parents=True, exist_ok=True)
    payload = {
        "model": model.state_dict(),
        "ema": ema.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scheduler": None if scheduler is None else scheduler.state_dict(),
        "scaler": None if scaler is None else scaler.state_dict(),
        "schedule": schedule.state_dict(),
        "mu": bank.mu.tolist(),
        "sigma": bank.sigma.tolist(),
        "layernorm_weight": None if bank.layernorm_weight is None else bank.layernorm_weight.tolist(),
        "layernorm_bias": None if bank.layernorm_bias is None else bank.layernorm_bias.tolist(),
        "checkpoint_hash": bank.checkpoint_hash,
        "dim": bank.dim,
        "step": step,
        "best_metric": best_metric,
        "config": cfg,
        "extra": extra or {},
    }
    tmp = output / "sampler.pt.tmp"
    torch.save(payload, tmp)
    os.replace(tmp, output / "sampler.pt")
    with (output / "sampler_config.json").open("w", encoding="utf-8") as handle:
        json.dump(cfg, handle, indent=2, sort_keys=True)
    log(TAG, f"saved sampler checkpoint -> {output} (step={step})")


def train_sampler_from_config(cfg: dict[str, Any]) -> Path:
    rank, local_rank, world_size, ddp_device = _setup_distributed(
        float(cfg.get("dist_timeout_min", 60.0))
    )
    is_main = rank == 0
    # Offset the seed per rank so every rank draws a different neighborhood episode:
    # DDP averages their grads, so the effective batch is batch_size * world_size.
    seed_everything(int(cfg.get("seed", 0)) + rank)
    device = ddp_device or str(cfg.get("device", "cuda" if torch.cuda.is_available() else "cpu"))
    bank_dir = Path(cfg["bank_dir"])
    output = Path(cfg.get("output", storage_root() / "models" / "cue" / "sampler"))
    if is_main:
        output.mkdir(parents=True, exist_ok=True)

    bank = load_bank(bank_dir)
    if bank.prior_index is None:
        raise RuntimeError("bank missing prior FAISS/numpy index")
    if bank.checkpoint_hash and cfg.get("joint_training_checkpoint"):
        from cue_training.sampler.bank import checkpoint_content_hash

        h = checkpoint_content_hash(cfg["joint_training_checkpoint"])
        if h != bank.checkpoint_hash:
            warn(
                TAG,
                f"bank checkpoint hash {bank.checkpoint_hash} != joint_training {h}; continuing anyway",
            )

    dim = bank.dim
    model = DiffusionSamplerModel(
        dim=dim,
        width=int(cfg.get("width", 1536)),
        depth=int(cfg.get("depth", 6)),
        time_dim=int(cfg.get("time_dim", 256)),
        conditioner_heads=int(cfg.get("conditioner_heads", 4)),
    ).to(device)
    ema = EMA(model, decay=float(cfg.get("ema_decay", 0.999)))
    if world_size > 1:
        # Construction broadcasts rank-0 weights, so the per-rank seed offset above does
        # not desync init. EMA/save/clip go through _unwrap to reach the real module.
        model = torch.nn.parallel.DistributedDataParallel(
            model,
            device_ids=[local_rank] if device.startswith("cuda") else None,
        )
    schedule = CosineNoiseSchedule(timesteps=int(cfg.get("timesteps", 1000)))
    schedule.to(device)

    lr = float(cfg.get("lr", 2e-4))
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=float(cfg.get("weight_decay", 0.01)))
    max_steps = int(cfg.get("max_steps", 20_000))
    warmup = int(cfg.get("warmup_steps", 500))
    def lr_lambda(step: int) -> float:
        if step < warmup:
            return (step + 1) / max(warmup, 1)
        return max(0.1, 1.0 - (step - warmup) / max(max_steps - warmup, 1))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    use_amp = bool(cfg.get("amp", device.startswith("cuda"))) and device.startswith("cuda")
    scaler = _make_scaler(use_amp)

    batch_size = int(cfg.get("batch_size", 64))
    k = int(cfg.get("k", 8))
    p_uncond = float(cfg.get("p_uncond", 0.15))
    lambda_cons = float(cfg.get("lambda_consistency", 0.1))
    lambda_nn = float(cfg.get("lambda_neighborhood", 0.05))
    lambda_ctr = float(cfg.get("lambda_contrastive", 0.05))
    log_every = int(cfg.get("log_every", 50))
    val_every = int(cfg.get("val_every", 1000))
    grad_clip = float(cfg.get("grad_clip", 1.0))

    prior_emb = np.asarray(bank.prior_train.embeddings, dtype=np.float32)
    prior_std = ((prior_emb - bank.mu) / bank.sigma).astype(np.float32)
    # Rebuild index on standardized vectors for neighborhood in train space.
    from cue_training.sampler.faiss_index import EmbeddingIndex

    prior_index_std = EmbeddingIndex(dim)
    prior_index_std.add(prior_std)

    sources = list(bank.prior_train.sources)
    if sources and len(sources) != prior_std.shape[0]:
        warn(TAG, f"{len(sources)} source labels for {prior_std.shape[0]} rows; sampling uniformly")
        sources = []
    anchor_weights = _source_weights(sources, float(cfg.get("source_balance_alpha", 1.0)))
    if is_main and anchor_weights is not None:
        mass = Counter()
        for source, weight in zip(sources, anchor_weights, strict=True):
            mass[source] += float(weight)
        log(TAG, "anchor mass by source: " + ", ".join(
            f"{s}={m:.3f}" for s, m in sorted(mass.items(), key=lambda kv: -kv[1])
        ))
    same_source_frac = float(cfg.get("condition_same_source_frac", 0.0))
    source_codes = np.unique(sources, return_inverse=True)[1] if sources else None
    if is_main and source_codes is not None and same_source_frac > 0:
        log(TAG, f"conditioning: {same_source_frac:.0%} of episodes restricted to one source")

    rng = np.random.default_rng(int(cfg.get("seed", 0)))
    best_metric = float("inf")
    start_step = 0
    resume = cfg.get("resume")
    if resume:
        payload = torch.load(Path(resume) / "sampler.pt", map_location=device, weights_only=False)
        _unwrap(model).load_state_dict(payload["model"])
        ema.load_state_dict(payload["ema"], model=_unwrap(model))
        optimizer.load_state_dict(payload["optimizer"])
        if payload.get("scheduler") and scheduler is not None:
            scheduler.load_state_dict(payload["scheduler"])
        if payload.get("scaler") and scaler is not None:
            scaler.load_state_dict(payload["scaler"])
        start_step = int(payload.get("step", 0))
        best_metric = float(payload.get("best_metric", best_metric))
        log(TAG, f"resumed from {resume} at step {start_step}")

    wandb_run = None
    if is_main and not cfg.get("no_wandb", False):
        try:
            import wandb

            wandb_run = wandb.init(
                project=cfg.get("wandb_project", "cue-unified"),
                entity=cfg.get("wandb_entity"),
                name=cfg.get("wandb_name", "sampler"),
                group=cfg.get("wandb_group"),
                config=cfg,
            )
        except Exception as exc:  # noqa: BLE001
            warn(TAG, f"wandb init failed: {exc}")

    model.train()
    for step in range(start_step, max_steps):
        z0, set_t, mask_t, drop_t = _sample_neighborhood_episode(
            prior_std,
            prior_index_std,
            batch_size=batch_size,
            k=k,
            rng=rng,
            p_uncond=p_uncond,
            weights=anchor_weights,
            source_codes=source_codes,
            same_source_frac=same_source_frac,
        )
        z0 = z0.to(device)
        set_t = set_t.to(device)
        mask_t = mask_t.to(device)
        drop_t = drop_t.to(device)
        t = torch.randint(0, schedule.timesteps, (batch_size,), device=device)

        optimizer.zero_grad(set_to_none=True)
        with _autocast(use_amp):
            zt, noise = schedule.q_sample(z0, t)
            eps_pred = model(zt, t, set_emb=set_t, mask=mask_t, drop_condition=drop_t)
            loss_diff = F.mse_loss(eps_pred, noise)
            z0_hat = schedule.predict_x0_from_eps(zt, t, eps_pred)
            loss_cons = (1.0 - F.cosine_similarity(z0_hat, z0, dim=-1)).mean()
            # neighborhood pull toward stop-grad set mean (conditional only)
            set_mean = (set_t * mask_t.unsqueeze(-1)).sum(dim=1) / mask_t.sum(dim=1).clamp_min(1).unsqueeze(-1)
            set_mean = set_mean.detach()
            cond_mask = (~drop_t) & (mask_t.sum(dim=1) > 0)
            if cond_mask.any():
                loss_nn = (1.0 - F.cosine_similarity(z0_hat[cond_mask], set_mean[cond_mask], dim=-1)).mean()
            else:
                loss_nn = z0_hat.sum() * 0.0
            loss_ctr = _infonce(z0_hat, z0, temperature=float(cfg.get("contrastive_temp", 0.1)))
            loss = (
                loss_diff
                + lambda_cons * loss_cons
                + lambda_nn * loss_nn
                + lambda_ctr * loss_ctr
            )

        scaler.scale(loss).backward()
        if grad_clip > 0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        scaler.step(optimizer)
        scaler.update()
        scheduler.step()
        ema.update(_unwrap(model))

        if is_main and step % log_every == 0:
            metrics = {
                "loss": float(loss.detach()),
                "loss_diff": float(loss_diff.detach()),
                "loss_cons": float(loss_cons.detach()),
                "loss_nn": float(loss_nn.detach()),
                "loss_ctr": float(loss_ctr.detach()),
                "lr": float(scheduler.get_last_lr()[0]),
                "step": step,
            }
            log(TAG, " ".join(f"{k}={v:.4f}" if isinstance(v, float) else f"{k}={v}" for k, v in metrics.items()))
            if wandb_run is not None:
                wandb_run.log(metrics, step=step)

        if val_every > 0 and step > 0 and step % val_every == 0:
            # Every rank validates: the denoise rows, the DDIM draws, and the decode set
            # are split across ranks and aggregated, so no rank sits idle long enough to
            # trip the collective timeout while another one generates.
            val = _validate(
                cfg, bank, ema.shadow, schedule, device, rank=rank, world_size=world_size
            )
            if not is_main:
                dist.barrier()
                continue
            # denoising_mse keeps falling while samples drift off the data manifold, so it
            # cannot select on its own; median_nn_dev is the metric that caught the drift.
            score = (
                float(val.get("denoising_mse", 0.0))
                - 0.1 * float(val.get("parse_rate", 0.0))
                + float(cfg.get("manifold_score_weight", 1.0))
                * float(np.nan_to_num(val.get("median_nn_dev", 0.0)))
            )
            log(TAG, f"val step={step} score={score:.4f} {val}")
            if wandb_run is not None:
                wandb_run.log({f"val/{k}": v for k, v in val.items()}, step=step)
            if score < best_metric:
                best_metric = score
                _save_sampler_checkpoint(
                    output / "best",
                    model=_unwrap(model),
                    ema=ema,
                    optimizer=optimizer,
                    scheduler=scheduler,
                    scaler=scaler,
                    schedule=schedule,
                    bank=bank,
                    cfg=cfg,
                    step=step,
                    best_metric=best_metric,
                    extra={"val": val},
                )
            _save_sampler_checkpoint(
                output / "last",
                model=_unwrap(model),
                ema=ema,
                optimizer=optimizer,
                scheduler=scheduler,
                scaler=scaler,
                schedule=schedule,
                bank=bank,
                cfg=cfg,
                step=step,
                best_metric=best_metric,
                extra={"val": val},
            )
            if world_size > 1:
                dist.barrier()

    if not is_main:
        _cleanup_distributed()
        return output
    _save_sampler_checkpoint(
        output / "last",
        model=_unwrap(model),
        ema=ema,
        optimizer=optimizer,
        scheduler=scheduler,
        scaler=scaler,
        schedule=schedule,
        bank=bank,
        cfg=cfg,
        step=max_steps,
        best_metric=best_metric,
    )
    if wandb_run is not None:
        wandb_run.finish()
    _cleanup_distributed()
    return output


@torch.no_grad()
def _validate(
    cfg: dict[str, Any],
    bank: EmbeddingBank,
    model: DiffusionSamplerModel,
    schedule: CosineNoiseSchedule,
    device: str,
    rank: int = 0,
    world_size: int = 1,
) -> dict[str, Any]:
    model.eval()
    schedule.to(device)
    n_val = min(int(cfg.get("val_denoise_n", 256)), max(1, bank.prior_dev.n or bank.prior_train.n))
    src = bank.prior_dev if bank.prior_dev.n else bank.prior_train
    emb = np.asarray(src.embeddings[:n_val], dtype=np.float32)
    local = emb[rank::world_size] if world_size > 1 else emb
    z0 = torch.from_numpy(((local - bank.mu) / bank.sigma).astype(np.float32)).to(device)
    mse = denoising_mse(model, schedule, z0) if z0.shape[0] else 0.0
    if world_size > 1:
        # Weight by row count: strided slices can differ in length by one.
        mse_sum, seen = _all_reduce_sum([mse * z0.shape[0], float(z0.shape[0])], device)
        mse = mse_sum / max(seen, 1.0)

    n_sample = int(cfg.get("val_sample_n", 64))
    per_rank = -(-n_sample // world_size)
    z_std = ddim_sample(
        model,
        schedule,
        n=per_rank,
        steps=int(cfg.get("ddim_steps", 50)),
        guidance_w=0.0,
        device=device,
    )
    if world_size > 1:
        # Each rank seeded differently, so these are distinct draws; equal per-rank counts
        # keep all_gather's shapes uniform, then trim back to the configured total.
        chunks = [torch.empty_like(z_std) for _ in range(world_size)]
        dist.all_gather(chunks, z_std.contiguous())
        z_std = torch.cat(chunks, dim=0)[:n_sample]
    samples = (z_std.cpu().numpy() * bank.sigma + bank.mu).astype(np.float32)
    if bool(cfg.get("project_manifold", False)):
        # Rollouts project onto the LN manifold before decoding, so scoring raw DDIM output
        # here measured a distribution inference never sees.
        samples = (
            project_layernorm_manifold(
                torch.from_numpy(samples),
                None if bank.layernorm_weight is None else torch.from_numpy(bank.layernorm_weight),
                None if bank.layernorm_bias is None else torch.from_numpy(bank.layernorm_bias),
            )
            .numpy()
            .astype(np.float32)
        )
    # Same samples and a fixed internal seed on every rank, so this agrees without a collective.
    dist_metrics = evaluate_samples_vs_bank(samples, bank)

    out: dict[str, Any] = {"denoising_mse": mse, **dist_metrics}
    # optional decode gate
    if cfg.get("val_decode", True) and cfg.get("joint_training_checkpoint"):
        decode_rows = samples[: int(cfg.get("val_decode_n", 16))]
        decode_kwargs = {
            "max_new_tokens": int(cfg.get("val_gen_max_tokens", 512)),
            "batch_size": int(cfg.get("val_decode_batch", 4)),
            "rank": rank,
            "world_size": world_size,
            "device": device,
        }
        dec = _load_decoder_everywhere(cfg["joint_training_checkpoint"], device, world_size)
        if dec is not None:
            parse = _decode_parse_sharded(dec, decode_rows, **decode_kwargs)
            if parse is not None:
                out["parse_rate"] = parse["parse_rate"]
                out["parse_mean_length"] = parse["mean_length"]
                out["parse_duplicate_rate"] = parse["duplicate_rate"]
            if cfg.get("refinement_checkpoint"):
                dec2 = _load_decoder_everywhere(cfg["refinement_checkpoint"], device, world_size)
                if dec2 is not None:
                    parse2 = _decode_parse_sharded(dec2, decode_rows, **decode_kwargs)
                    if parse2 is not None:
                        out["parse_rate_refinement"] = parse2["parse_rate"]

    baselines = (bank.report or {}).get("baselines") or {}
    thresholds = cfg.get("acceptance") or {}
    if thresholds:
        report = acceptance_report(
            {
                "parse_rate_joint_training": out.get("parse_rate"),
                "parse_rate_refinement": out.get("parse_rate_refinement"),
                "sliced_wasserstein": out.get("sliced_wasserstein"),
                "rff_mmd": out.get("rff_mmd"),
                "memorization_ratio": out.get("memorization_ratio"),
                "bias_gap": out.get("bias_gap"),
                "diversity": out.get("diversity"),
            },
            thresholds,
            baselines,
        )
        out["acceptance_pass"] = report["pass"]
        out["acceptance"] = report
    return out
