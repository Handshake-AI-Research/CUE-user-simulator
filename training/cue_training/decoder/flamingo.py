"""Flamingo-style GATED XATTN-DENSE residual blocks (Alayrac et al., 2022)."""

from __future__ import annotations

import torch
import torch.nn as nn


class GatedXAttnDense(nn.Module):
    """Cross-attend LM hidden states to persona memory, then gated dense FFN.

    Gates are learnable scalars initialized to 0 so tanh(alpha)=0 at start.
    Block math runs in fp32: bf16 MHA/LayerNorm is unstable, and ``0 * nan``
    would still poison the residual when gates are zero.
    """

    def __init__(
        self,
        hidden_dim: int,
        num_heads: int = 8,
        ff_mult: int = 4,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.ln_xattn = nn.LayerNorm(hidden_dim)
        self.xattn = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.alpha_xattn = nn.Parameter(torch.zeros(1))
        self.ln_dense = nn.LayerNorm(hidden_dim)
        self.dense = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * ff_mult),
            nn.GELU(),
            nn.Linear(hidden_dim * ff_mult, hidden_dim),
        )
        self.alpha_dense = nn.Parameter(torch.zeros(1))

    def forward(self, hidden: torch.Tensor, persona_tokens: torch.Tensor) -> torch.Tensor:
        """
        hidden: [B, T, H]
        persona_tokens: [B, M, H]
        """

        orig_dtype = hidden.dtype
        h = hidden.float()
        kv = persona_tokens.float()
        attended, _ = self.xattn(self.ln_xattn(h), kv, kv)
        # Sanitize before gating: IEEE 0*inf/nan would still poison y = x + 0*a.
        attended = torch.nan_to_num(attended, nan=0.0, posinf=0.0, neginf=0.0)
        dense_out = torch.nan_to_num(
            self.dense(self.ln_dense(h)), nan=0.0, posinf=0.0, neginf=0.0
        )
        h = h + torch.tanh(self.alpha_xattn).float() * attended
        h = h + torch.tanh(self.alpha_dense).float() * dense_out
        return h.to(dtype=orig_dtype)
