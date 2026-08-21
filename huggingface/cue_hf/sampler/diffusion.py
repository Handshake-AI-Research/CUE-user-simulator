"""Cosine schedule, set conditioner, and residual-MLP epsilon denoiser."""

from __future__ import annotations

import math
from typing import Any

import torch
import torch.nn.functional as F
from torch import nn


def cosine_beta_schedule(timesteps: int, s: float = 0.008) -> torch.Tensor:
    """Improved DDPM cosine schedule (Nichol & Dhariwal)."""

    steps = timesteps + 1
    x = torch.linspace(0, timesteps, steps, dtype=torch.float64)
    alphas_cumprod = torch.cos(((x / timesteps) + s) / (1 + s) * math.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1.0 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return torch.clamp(betas, 1e-5, 0.999).float()


class CosineNoiseSchedule:
    """Closed-form q(z_t|z_0) helpers for epsilon-prediction sampling."""

    def __init__(self, timesteps: int = 1000, s: float = 0.008) -> None:
        self.timesteps = int(timesteps)
        betas = cosine_beta_schedule(self.timesteps, s=s)
        alphas = 1.0 - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)
        alphas_cumprod_prev = F.pad(alphas_cumprod[:-1], (1, 0), value=1.0)
        self.betas = betas
        self.alphas = alphas
        self.alphas_cumprod = alphas_cumprod
        self.alphas_cumprod_prev = alphas_cumprod_prev
        self.sqrt_alphas_cumprod = torch.sqrt(alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0 - alphas_cumprod)
        self.sqrt_recip_alphas_cumprod = torch.sqrt(1.0 / alphas_cumprod)
        self.sqrt_recipm1_alphas_cumprod = torch.sqrt(1.0 / alphas_cumprod - 1.0)

    def to(self, device: torch.device | str) -> CosineNoiseSchedule:
        for name in (
            "betas",
            "alphas",
            "alphas_cumprod",
            "alphas_cumprod_prev",
            "sqrt_alphas_cumprod",
            "sqrt_one_minus_alphas_cumprod",
            "sqrt_recip_alphas_cumprod",
            "sqrt_recipm1_alphas_cumprod",
        ):
            setattr(self, name, getattr(self, name).to(device))
        return self

    def _gather(self, values: torch.Tensor, t: torch.Tensor, *, like: torch.Tensor) -> torch.Tensor:
        out = values.gather(0, t.long().clamp(0, self.timesteps - 1))
        while out.ndim < like.ndim:
            out = out.unsqueeze(-1)
        return out.to(dtype=like.dtype)

    def predict_x0_from_eps(self, zt: torch.Tensor, t: torch.Tensor, eps: torch.Tensor) -> torch.Tensor:
        sqrt_recip = self._gather(self.sqrt_recip_alphas_cumprod, t, like=zt)
        sqrt_recipm1 = self._gather(self.sqrt_recipm1_alphas_cumprod, t, like=zt)
        return sqrt_recip * zt - sqrt_recipm1 * eps

    def state_dict(self) -> dict[str, Any]:
        return {"timesteps": self.timesteps, "betas": self.betas.detach().cpu()}

    @classmethod
    def from_state_dict(cls, state: dict[str, Any]) -> CosineNoiseSchedule:
        obj = cls(timesteps=int(state["timesteps"]))
        if "betas" in state:
            betas = state["betas"]
            if not torch.is_tensor(betas):
                betas = torch.tensor(betas)
            obj.betas = betas.float()
            obj.alphas = 1.0 - obj.betas
            obj.alphas_cumprod = torch.cumprod(obj.alphas, dim=0)
            obj.alphas_cumprod_prev = F.pad(obj.alphas_cumprod[:-1], (1, 0), value=1.0)
            obj.sqrt_alphas_cumprod = torch.sqrt(obj.alphas_cumprod)
            obj.sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0 - obj.alphas_cumprod)
            obj.sqrt_recip_alphas_cumprod = torch.sqrt(1.0 / obj.alphas_cumprod)
            obj.sqrt_recipm1_alphas_cumprod = torch.sqrt(1.0 / obj.alphas_cumprod - 1.0)
        return obj


def timestep_embedding(timesteps: torch.Tensor, dim: int, max_period: int = 10_000) -> torch.Tensor:
    half = dim // 2
    freqs = torch.exp(
        -math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32, device=timesteps.device) / half
    )
    args = timesteps.float().unsqueeze(1) * freqs.unsqueeze(0)
    emb = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
    if dim % 2 == 1:
        emb = F.pad(emb, (0, 1))
    return emb


class SetConditioner(nn.Module):
    """Permutation-invariant attention pooling over a variable-size set."""

    def __init__(self, dim: int, hidden: int | None = None, num_heads: int = 4) -> None:
        super().__init__()
        hidden = hidden or dim
        self.dim = dim
        self.proj = nn.Linear(dim, hidden)
        self.query = nn.Parameter(torch.randn(1, 1, hidden) * 0.02)
        self.attn = nn.MultiheadAttention(hidden, num_heads, batch_first=True)
        self.out = nn.Linear(hidden, dim)
        self.null = nn.Parameter(torch.zeros(dim))

    def forward(
        self,
        set_emb: torch.Tensor | None,
        mask: torch.Tensor | None = None,
        *,
        drop_condition: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        set_emb: [B, K, D] or None (all-null).
        mask: [B, K] True = valid member.
        drop_condition: [B] bool — True means use null (CFG dropout).
        """

        if set_emb is None:
            batch = 1 if drop_condition is None else int(drop_condition.shape[0])
            return self.null.unsqueeze(0).expand(batch, -1)

        batch, _, _ = set_emb.shape
        x = self.proj(set_emb)
        q = self.query.expand(batch, -1, -1)
        key_padding = None
        if mask is not None:
            key_padding = ~mask.bool()
            # MultiheadAttention requires at least one valid key; force first if empty.
            empty = mask.sum(dim=-1) == 0
            if empty.any():
                key_padding = key_padding.clone()
                key_padding[empty, 0] = False
        pooled, _ = self.attn(q, x, x, key_padding_mask=key_padding, need_weights=False)
        c = self.out(pooled.squeeze(1))
        if drop_condition is not None:
            drop = drop_condition.bool().unsqueeze(-1)
            c = torch.where(drop, self.null.unsqueeze(0).expand_as(c), c)
        elif mask is not None:
            empty = (mask.sum(dim=-1) == 0).unsqueeze(-1)
            c = torch.where(empty, self.null.unsqueeze(0).expand_as(c), c)
        return c


class FiLMBlock(nn.Module):
    def __init__(self, dim: int, cond_dim: int) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fc1 = nn.Linear(dim, dim * 4)
        self.fc2 = nn.Linear(dim * 4, dim)
        self.film = nn.Linear(cond_dim, dim * 2)

    def forward(self, x: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        h = self.norm(x)
        scale, shift = self.film(cond).chunk(2, dim=-1)
        h = h * (1.0 + scale) + shift
        h = self.fc2(F.gelu(self.fc1(h)))
        return x + h


class EmbeddingDenoiser(nn.Module):
    """Residual MLP epsilon predictor with sinusoidal t + FiLM conditioning."""

    def __init__(
        self,
        dim: int = 1024,
        width: int = 1536,
        depth: int = 6,
        time_dim: int = 256,
        cond_dim: int | None = None,
    ) -> None:
        super().__init__()
        cond_dim = cond_dim or dim
        self.dim = dim
        self.time_dim = time_dim
        self.inp = nn.Linear(dim, width)
        self.time_mlp = nn.Sequential(
            nn.Linear(time_dim, width),
            nn.SiLU(),
            nn.Linear(width, width),
        )
        self.cond_proj = nn.Linear(cond_dim, width)
        self.blocks = nn.ModuleList([FiLMBlock(width, width) for _ in range(depth)])
        self.out_norm = nn.LayerNorm(width)
        self.out = nn.Linear(width, dim)

    def forward(self, zt: torch.Tensor, t: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        t_emb = timestep_embedding(t, self.time_dim)
        cond_h = self.time_mlp(t_emb) + self.cond_proj(cond)
        h = self.inp(zt)
        for block in self.blocks:
            h = block(h, cond_h)
        return self.out(self.out_norm(h))


class DiffusionSamplerModel(nn.Module):
    """Denoiser + set conditioner package used for sampling."""

    def __init__(
        self,
        dim: int = 1024,
        width: int = 1536,
        depth: int = 6,
        time_dim: int = 256,
        conditioner_heads: int = 4,
    ) -> None:
        super().__init__()
        self.dim = dim
        self.conditioner = SetConditioner(dim, hidden=width, num_heads=conditioner_heads)
        self.denoiser = EmbeddingDenoiser(dim=dim, width=width, depth=depth, time_dim=time_dim, cond_dim=dim)

    def forward(
        self,
        zt: torch.Tensor,
        t: torch.Tensor,
        set_emb: torch.Tensor | None = None,
        mask: torch.Tensor | None = None,
        drop_condition: torch.Tensor | None = None,
        cond: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if cond is None:
            cond = self.conditioner(set_emb, mask, drop_condition=drop_condition)
        return self.denoiser(zt, t, cond)
