"""Layer-wise attention pooling over transformer hidden states."""

from __future__ import annotations

import torch
import torch.nn as nn


class LayerwiseAttention(nn.Module):
    def __init__(self, num_layers: int) -> None:
        super().__init__()
        self.num_layers = num_layers
        self.weights = nn.Parameter(torch.zeros(num_layers))
        self.gamma = nn.Parameter(torch.ones(1))

    def forward(self, hidden_states: tuple[torch.Tensor, ...] | list[torch.Tensor]) -> torch.Tensor:
        if len(hidden_states) != self.num_layers:
            raise ValueError(f"expected {self.num_layers} layers, got {len(hidden_states)}")
        w = torch.softmax(self.weights, dim=0)
        stacked = torch.stack(list(hidden_states), dim=0)  # [L,B,T,H]
        combined = (w.view(-1, 1, 1, 1) * stacked).sum(dim=0)
        return self.gamma * combined
