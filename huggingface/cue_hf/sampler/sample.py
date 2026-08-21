"""DDIM sampling with classifier-free guidance over CUE embeddings."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch

from cue_hf.sampler.diffusion import CosineNoiseSchedule, DiffusionSamplerModel

SAMPLER_NAME = "sampler.pt"


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
    timesteps = torch.linspace(schedule.timesteps - 1, 0, steps, device=device).long()

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
            t_next = torch.full((n,), int(timesteps[i + 1].item()), device=device, dtype=torch.long)
            alpha_next = schedule._gather(schedule.alphas_cumprod, t_next, like=zt)
            # DDIM (eta=0): z_{t-1} = sqrt(a_next)*x0 + sqrt(1-a_next)*eps
            zt = torch.sqrt(alpha_next) * x0 + torch.sqrt(1.0 - alpha_next) * eps
        else:
            zt = x0
    return zt


@dataclass
class CueSampler:
    """Loaded diffusion prior over CUE bottlenecks, with its standardization stats."""

    model: DiffusionSamplerModel
    schedule: CosineNoiseSchedule
    mu: torch.Tensor
    sigma: torch.Tensor
    layernorm_weight: torch.Tensor | None
    layernorm_bias: torch.Tensor | None
    config: dict[str, Any]

    @property
    def dim(self) -> int:
        return int(self.model.dim)

    def to(self, device: str | torch.device) -> CueSampler:
        self.model.to(device)
        self.schedule.to(device)
        self.mu = self.mu.to(device)
        self.sigma = self.sigma.to(device)
        if self.layernorm_weight is not None:
            self.layernorm_weight = self.layernorm_weight.to(device)
        if self.layernorm_bias is not None:
            self.layernorm_bias = self.layernorm_bias.to(device)
        return self

    @torch.no_grad()
    def sample(
        self,
        *,
        n: int = 8,
        condition: np.ndarray | torch.Tensor | list | None = None,
        steps: int | None = None,
        guidance_w: float | None = None,
        project_manifold: bool | None = None,
        seed: int = 0,
    ) -> torch.Tensor:
        """Return unstandardized CUE embeddings [n, D]."""

        device = self.mu.device
        steps = int(steps if steps is not None else self.config.get("ddim_steps", 50))
        if guidance_w is None:
            guidance_w = float(self.config.get("guidance_w", 1.5))
        if project_manifold is None:
            project_manifold = bool(self.config.get("project_manifold", False))

        set_emb = None
        mask = None
        if condition is not None:
            cond = as_condition_tensor(condition, device=device)
            set_emb = (cond - self.mu) / self.sigma
            mask = torch.ones(set_emb.shape[0], set_emb.shape[1], dtype=torch.bool, device=device)
            if not torch.isfinite(set_emb).all():
                raise ValueError("condition embeddings contain non-finite values")
            if set_emb.shape[-1] != self.dim:
                raise ValueError(f"expected dim={self.dim}, got {set_emb.shape[-1]}")

        generator = torch.Generator(device=device)
        generator.manual_seed(int(seed))
        z_std = ddim_sample(
            self.model,
            self.schedule,
            n=n,
            steps=steps,
            guidance_w=float(guidance_w) if set_emb is not None else 0.0,
            set_emb=set_emb,
            mask=mask,
            generator=generator,
            device=device,
        )
        z = z_std * self.sigma + self.mu
        if project_manifold:
            z = project_layernorm_manifold(z, self.layernorm_weight, self.layernorm_bias)
        return z


def load_sampler(
    checkpoint: str | Path,
    *,
    device: str = "cpu",
    use_ema: bool = True,
) -> CueSampler:
    """Load ``sampler.pt`` from a CUE sampler directory (or a direct file path)."""

    path = Path(checkpoint)
    if path.is_dir():
        path = path / SAMPLER_NAME
    if not path.is_file():
        raise FileNotFoundError(f"sampler weights not found: {path}")
    payload = torch.load(path, map_location=device, weights_only=False)
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
    schedule = CosineNoiseSchedule.from_state_dict(payload["schedule"]).to(device)
    ln_w = payload.get("layernorm_weight")
    ln_b = payload.get("layernorm_bias")
    return CueSampler(
        model=model,
        schedule=schedule,
        mu=torch.tensor(payload["mu"], device=device, dtype=torch.float32),
        sigma=torch.tensor(payload["sigma"], device=device, dtype=torch.float32),
        layernorm_weight=(torch.tensor(ln_w, device=device, dtype=torch.float32) if ln_w is not None else None),
        layernorm_bias=(torch.tensor(ln_b, device=device, dtype=torch.float32) if ln_b is not None else None),
        config=dict(cfg),
    )


"""Keys ``load_sampler`` reads, plus provenance scalars. Everything else is training state."""
INFERENCE_KEYS = (
    "schedule",
    "mu",
    "sigma",
    "layernorm_weight",
    "layernorm_bias",
    "dim",
    "config",
    "checkpoint_hash",
    "step",
)


def slim_sampler_payload(payload: dict[str, Any]) -> dict[str, Any]:
    """Drop optimizer/scheduler/scaler state and the duplicate weight copy.

    Training checkpoints hold ``model``, an EMA shadow, and optimizer moments. Inference
    uses the EMA weights, so those are promoted into ``model`` and the rest is dropped:
    one copy that loads identically whether or not ``use_ema`` is set.
    """

    ema = payload.get("ema")
    if isinstance(ema, dict) and "shadow" in ema:
        weights = ema["shadow"]
    elif "model" in payload:
        weights = payload["model"]
    else:
        raise KeyError("sampler payload has neither ema['shadow'] nor model weights")
    slim: dict[str, Any] = {"model": weights}
    slim.update({key: payload[key] for key in INFERENCE_KEYS if key in payload})
    return slim


def slim_sampler_file(source: str | Path, destination: str | Path) -> Path:
    """Rewrite a ``sampler.pt`` with only the tensors inference needs."""

    source, destination = Path(source), Path(destination)
    if source.is_dir():
        source = source / SAMPLER_NAME
    payload = torch.load(source, map_location="cpu", mmap=True, weights_only=False)
    torch.save(slim_sampler_payload(payload), destination)
    return destination


def as_condition_tensor(
    condition: np.ndarray | torch.Tensor | list,
    *,
    device: str | torch.device,
) -> torch.Tensor:
    if torch.is_tensor(condition):
        t = condition.detach().float()
    else:
        t = torch.from_numpy(np.asarray(condition, dtype=np.float32))
    if t.ndim == 1:
        t = t.unsqueeze(0).unsqueeze(0)  # [1,1,D]
    elif t.ndim == 2:
        t = t.unsqueeze(0)  # [1,K,D]
    elif t.ndim != 3:
        raise ValueError(f"condition embeddings must be [D], [K,D], or [B,K,D]; got {tuple(t.shape)}")
    return t.to(device)
