"""User-turn query attends over full system/assistant token states."""

from __future__ import annotations

import torch
from torch import nn


class SystemContextFusion(nn.Module):
    """Cross-attend a pooled user query over a system token sequence."""

    def __init__(self, hidden_dim: int, num_heads: int = 8, dropout: float = 0.1) -> None:
        super().__init__()
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.norm = nn.LayerNorm(hidden_dim)
        self.gate = nn.Parameter(torch.zeros(1))
        self.no_system = nn.Parameter(torch.zeros(1, 1, hidden_dim))

    def forward(
        self,
        user_query: torch.Tensor,
        system_tokens: torch.Tensor | None,
        system_mask: torch.Tensor | None,
        has_system: torch.Tensor,
    ) -> torch.Tensor:
        """
        user_query: [B, H]
        system_tokens: [B, S, H] or None
        system_mask: [B, S] True=valid
        has_system: [B] bool
        """

        batch = user_query.size(0)
        if system_tokens is None:
            system_tokens = self.no_system.expand(batch, -1, -1)
            key_padding = torch.zeros(batch, 1, dtype=torch.bool, device=user_query.device)
        elif system_mask is None:
            key_padding = torch.zeros(
                system_tokens.size(0),
                system_tokens.size(1),
                dtype=torch.bool,
                device=user_query.device,
            )
        else:
            # MultiheadAttention key_padding_mask: True = ignore
            key_padding = ~system_mask.bool()

        # Rows with no valid keys make softmax(all -inf) → NaN; skip those in MHA.
        valid_keys = ~key_padding
        use_system = has_system.bool() & valid_keys.any(dim=-1)
        attended = torch.zeros_like(user_query)
        if use_system.any():
            idx = use_system.nonzero(as_tuple=False).squeeze(-1)
            query = user_query[idx].unsqueeze(1)
            att, _ = self.cross_attn(
                query,
                system_tokens[idx],
                system_tokens[idx],
                key_padding_mask=key_padding[idx],
            )
            attended[idx] = att.squeeze(1)
        # Guard IEEE 0*nan when tanh(gate)≈0.
        attended = torch.nan_to_num(attended, nan=0.0, posinf=0.0, neginf=0.0)
        gated = user_query + torch.tanh(self.gate) * attended
        fused = self.norm(gated)
        return torch.where(use_system.unsqueeze(-1), fused, user_query)
