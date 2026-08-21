"""DDIM sampling with classifier-free guidance over cue embeddings."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F

from cue_training.data.schema import parse_manual
from cue_training.sampler.diffusion import CosineNoiseSchedule, DiffusionSamplerModel
from cue_training.utils.log import log

TAG = "sampler.sample"


def project_layernorm_manifold(
    z: torch.Tensor,
    weight: torch.Tensor | None,
    bias: torch.Tensor | None,
    eps: float = 1e-5,
) -> torch.Tensor:
    """Exact re-application of encoder bottleneck LayerNorm affine."""

    if weight is None or bias is None:
        # Unit-sphere-ish recenter/rescale fallback.
        mean = z.mean(dim=-1, keepdim=True)
        var = z.var(dim=-1, keepdim=True, unbiased=False)
        return (z - mean) / torch.sqrt(var + eps)
    # Invert y = gamma * x_hat + beta  =>  x_hat = (y - beta) / gamma
    x_hat = (z - bias) / weight.clamp_min(eps)
    # Re-center/re-scale then reapply affine (projects onto LN manifold).
    mean = x_hat.mean(dim=-1, keepdim=True)
    var = x_hat.var(dim=-1, keepdim=True, unbiased=False)
    x_hat = (x_hat - mean) / torch.sqrt(var + eps)
    return x_hat * weight + bias


@torch.no_grad()
def ddim_sample(
    model: DiffusionSamplerModel,
    schedule: CosineNoiseSchedule,
    *,
    n: int,
    steps: int = 50,
    guidance_w: float = 1.5,
    set_emb: torch.Tensor | None = None,
    mask: torch.Tensor | None = None,
    generator: torch.Generator | None = None,
    device: torch.device | str = "cpu",
) -> torch.Tensor:
    """
    Sample standardized embeddings via DDIM + CFG.

    set_emb: [B, K, D] standardized condition set (or None for unconditional).
    guidance_w: 0 / no set -> unconditional; typical 1–3 for conditional.
    """

    model.eval()
    device = torch.device(device)
    dim = model.dim
    if set_emb is not None:
        batch = set_emb.shape[0]
        if n != batch:
            # allow n to override only when broadcasting a single set
            if batch == 1 and n > 1:
                set_emb = set_emb.expand(n, -1, -1)
                if mask is not None:
                    mask = mask.expand(n, -1)
            else:
                n = batch
    zt = torch.randn(n, dim, device=device, generator=generator)
    timesteps = torch.linspace(
        schedule.timesteps - 1, 0, steps, device=device
    ).long()

    cond = None
    null = model.conditioner.null.unsqueeze(0).expand(n, -1)
    if set_emb is not None and guidance_w != 0:
        cond = model.conditioner(set_emb, mask)

    for i, t_scalar in enumerate(timesteps):
        t = torch.full((n,), int(t_scalar.item()), device=device, dtype=torch.long)
        eps_uncond = model.denoiser(zt, t, null)
        if cond is None or guidance_w == 0:
            eps = eps_uncond
        else:
            eps_cond = model.denoiser(zt, t, cond)
            eps = eps_uncond + guidance_w * (eps_cond - eps_uncond)
        x0 = schedule.predict_x0_from_eps(zt, t, eps)
        if i + 1 < len(timesteps):
            t_next = torch.full(
                (n,), int(timesteps[i + 1].item()), device=device, dtype=torch.long
            )
            alpha_next = schedule._gather(schedule.alphas_cumprod, t_next, like=zt)
            alpha_t = schedule._gather(schedule.alphas_cumprod, t, like=zt)
            # DDIM (eta=0): z_{t-1} = sqrt(a_next)*x0 + sqrt(1-a_next)*eps
            zt = torch.sqrt(alpha_next) * x0 + torch.sqrt(1.0 - alpha_next) * eps
            # numerical guard
            _ = alpha_t
        else:
            zt = x0
    return zt


def load_sampler(
    checkpoint: str | Path,
    *,
    device: str = "cpu",
    use_ema: bool = True,
) -> tuple[DiffusionSamplerModel, CosineNoiseSchedule, dict[str, Any]]:
    checkpoint = Path(checkpoint)
    payload = torch.load(checkpoint / "sampler.pt", map_location=device, weights_only=False)
    cfg = payload.get("config") or {}
    dim = int(cfg.get("dim", payload.get("dim", 1024)))
    model = DiffusionSamplerModel(
        dim=dim,
        width=int(cfg.get("width", 1536)),
        depth=int(cfg.get("depth", 6)),
        time_dim=int(cfg.get("time_dim", 256)),
        conditioner_heads=int(cfg.get("conditioner_heads", 4)),
    ).to(device)
    if use_ema and "ema" in payload:
        model.load_state_dict(payload["ema"]["shadow"])
    else:
        model.load_state_dict(payload["model"])
    model.eval()
    schedule = CosineNoiseSchedule.from_state_dict(payload["schedule"])
    schedule.to(device)
    return model, schedule, payload


def sample(
    checkpoint: str | Path,
    *,
    n: int = 8,
    steps: int | None = None,
    guidance_w: float | None = None,
    condition_embeddings: np.ndarray | torch.Tensor | list | None = None,
    device: str | None = None,
    project_manifold: bool | None = None,
    reject_decode: bool = False,
    max_reject_attempts: int = 3,
    decoder_checkpoint: str | Path | None = None,
    seed: int = 0,
) -> dict[str, Any]:
    """High-level sample API returning unstandardized embeddings (+ optional texts)."""

    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    model, schedule, payload = load_sampler(checkpoint, device=device, use_ema=True)
    cfg = payload.get("config") or {}
    steps = int(steps if steps is not None else cfg.get("ddim_steps", 50))
    guidance_w = float(guidance_w if guidance_w is not None else cfg.get("guidance_w", 1.5))
    if project_manifold is None:
        project_manifold = bool(cfg.get("project_manifold", False))

    mu = torch.tensor(payload["mu"], device=device, dtype=torch.float32)
    sigma = torch.tensor(payload["sigma"], device=device, dtype=torch.float32)
    ln_w = payload.get("layernorm_weight")
    ln_b = payload.get("layernorm_bias")
    ln_w_t = (
        torch.tensor(ln_w, device=device, dtype=torch.float32) if ln_w is not None else None
    )
    ln_b_t = (
        torch.tensor(ln_b, device=device, dtype=torch.float32) if ln_b is not None else None
    )

    set_emb = None
    mask = None
    if condition_embeddings is not None:
        cond = _as_condition_tensor(condition_embeddings, device=device)
        # standardize
        set_emb = (cond - mu) / sigma
        mask = torch.ones(set_emb.shape[0], set_emb.shape[1], dtype=torch.bool, device=device)
        # validate
        if not torch.isfinite(set_emb).all():
            raise ValueError("condition embeddings contain non-finite values")
        if set_emb.shape[-1] != model.dim:
            raise ValueError(f"expected dim={model.dim}, got {set_emb.shape[-1]}")

    g = torch.Generator(device=device)
    g.manual_seed(int(seed))

    attempts = 0
    texts: list[str] | None = None
    manuals: list[Any] | None = None
    while True:
        attempts += 1
        z_std = ddim_sample(
            model,
            schedule,
            n=n,
            steps=steps,
            guidance_w=guidance_w if set_emb is not None else 0.0,
            set_emb=set_emb,
            mask=mask,
            generator=g,
            device=device,
        )
        z = z_std * sigma + mu
        if project_manifold:
            z = project_layernorm_manifold(z, ln_w_t, ln_b_t)

        if not reject_decode:
            break
        if decoder_checkpoint is None:
            raise ValueError("reject_decode requires decoder_checkpoint")
        from cue_training.infer.export import load_checkpoint
        from cue_training.data.schema import COMMAND_BLOCK_SENTINEL

        dec_model = load_checkpoint(decoder_checkpoint, device=device)
        tok = dec_model.decoder.tokenizer
        sentinel = tok(
            COMMAND_BLOCK_SENTINEL, add_special_tokens=False, return_tensors="pt"
        )["input_ids"][0].to(device)
        texts = dec_model.decoder.generate(
            bottleneck=z, sentinel_ids=sentinel, max_new_tokens=int(cfg.get("val_gen_max_tokens", 512))
        )
        manuals = [parse_manual(t) for t in texts]
        ok = sum(m is not None for m in manuals)
        log(TAG, f"reject attempt {attempts}: parseable {ok}/{n}")
        if ok == n or attempts >= max_reject_attempts:
            break
        # resample only failures next? keep simple: full resample
        g.manual_seed(int(seed) + attempts)

    return {
        "embeddings": z.detach().float().cpu().numpy(),
        "embeddings_standardized": z_std.detach().float().cpu().numpy(),
        "texts": texts,
        "manuals": manuals,
        "attempts": attempts,
        "config": {
            "steps": steps,
            "guidance_w": guidance_w if set_emb is not None else 0.0,
            "project_manifold": project_manifold,
            "n": n,
        },
    }


def _as_condition_tensor(
    condition_embeddings: np.ndarray | torch.Tensor | list,
    *,
    device: str,
) -> torch.Tensor:
    if torch.is_tensor(condition_embeddings):
        t = condition_embeddings.detach().float()
    else:
        arr = np.asarray(condition_embeddings, dtype=np.float32)
        t = torch.from_numpy(arr)
    if t.ndim == 1:
        t = t.unsqueeze(0).unsqueeze(0)  # [1,1,D]
    elif t.ndim == 2:
        t = t.unsqueeze(0)  # [1,K,D]
    elif t.ndim != 3:
        raise ValueError(f"condition embeddings must be [D], [K,D], or [B,K,D]; got {tuple(t.shape)}")
    return t.to(device)
