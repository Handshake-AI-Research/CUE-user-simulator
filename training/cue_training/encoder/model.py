"""Hierarchical session encoder producing a deterministic bottleneck."""

from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn
from transformers import AutoModel, AutoTokenizer

from cue_training.encoder.layer_pool import LayerwiseAttention
from cue_training.encoder.system_fusion import SystemContextFusion
from cue_training.encoder.truncate import truncate_text_head_tail


def mean_pool(hidden: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    mask = attention_mask.unsqueeze(-1).to(hidden.dtype)
    return (hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1.0)


class HierarchicalSessionEncoder(nn.Module):
    """CLS-style session Transformer over fused user-turn vectors."""

    def __init__(
        self,
        input_dim: int,
        session_dim: int = 1024,
        num_layers: int = 4,
        num_heads: int = 8,
        dropout: float = 0.1,
        max_turns: int = 64,
    ) -> None:
        super().__init__()
        self.session_dim = session_dim
        self.input_proj = nn.Identity() if input_dim == session_dim else nn.Linear(input_dim, session_dim)
        self.cls = nn.Parameter(torch.randn(1, 1, session_dim) * 0.02)
        self.pos = nn.Embedding(max_turns + 1, session_dim)
        layer = nn.TransformerEncoderLayer(
            d_model=session_dim,
            nhead=num_heads,
            dim_feedforward=session_dim * 4,
            dropout=dropout,
            batch_first=True,
            activation="gelu",
        )
        self.transformer = nn.TransformerEncoder(
            layer, num_layers=num_layers, enable_nested_tensor=False
        )
        self.norm = nn.LayerNorm(session_dim)

    def forward(self, turn_embeds: torch.Tensor, turn_mask: torch.Tensor) -> torch.Tensor:
        """
        turn_embeds: [B, T, H]
        turn_mask: [B, T] True = padding
        """

        x = self.input_proj(turn_embeds)
        batch, turns, _ = x.shape
        cls = self.cls.expand(batch, -1, -1)
        seq = torch.cat([cls, x], dim=1)
        positions = torch.arange(turns + 1, device=x.device).unsqueeze(0)
        seq = seq + self.pos(positions)
        pad = torch.cat(
            [torch.zeros(batch, 1, dtype=torch.bool, device=x.device), turn_mask],
            dim=1,
        )
        encoded = self.transformer(seq, src_key_padding_mask=pad)
        return self.norm(encoded[:, 0])


class UnifiedEncoder(nn.Module):
    """Trainable user e5 + frozen system e5 → session bottleneck."""

    def __init__(
        self,
        *,
        user_model_name: str = "nomic-ai/modernbert-embed-base",
        system_model_name: str = "nomic-ai/modernbert-embed-base",
        bottleneck_dim: int = 1024,
        session_dim: int = 1024,
        session_layers: int = 4,
        session_heads: int = 8,
        max_turns: int = 64,
        max_length: int = 256,
        dropout: float = 0.1,
        query_prefix: str = "search_query: ",
        passage_prefix: str = "search_document: ",
    ) -> None:
        super().__init__()
        self.user_model_name = user_model_name
        self.system_model_name = system_model_name
        self.query_prefix = query_prefix
        self.passage_prefix = passage_prefix
        self.bottleneck_dim = bottleneck_dim
        self.max_turns = max_turns
        self.max_length = max_length

        self.user_tokenizer = AutoTokenizer.from_pretrained(
            user_model_name, clean_up_tokenization_spaces=False
        )
        self.system_tokenizer = AutoTokenizer.from_pretrained(
            system_model_name, clean_up_tokenization_spaces=False
        )
        # low_cpu_mem_usage=False: keep weights off the meta device so a later model.to(device)
        # doesn't fail with "Cannot copy out of meta tensor".
        self.user_encoder = AutoModel.from_pretrained(user_model_name, low_cpu_mem_usage=False)
        self.system_encoder = AutoModel.from_pretrained(system_model_name, low_cpu_mem_usage=False)
        for param in self.system_encoder.parameters():
            param.requires_grad_(False)
        self.system_encoder.eval()

        hidden = self.user_encoder.config.hidden_size
        num_layers = int(self.user_encoder.config.num_hidden_layers) + 1  # embeddings + layers
        self.layer_pool = LayerwiseAttention(num_layers)
        self.system_fusion = SystemContextFusion(hidden, num_heads=session_heads, dropout=dropout)
        self.session_encoder = HierarchicalSessionEncoder(
            input_dim=hidden,
            session_dim=session_dim,
            num_layers=session_layers,
            num_heads=session_heads,
            dropout=dropout,
            max_turns=max_turns,
        )
        self.bottleneck = nn.Sequential(
            nn.Linear(session_dim, bottleneck_dim),
            nn.LayerNorm(bottleneck_dim),
        )
        self.vicreg_proj = nn.Sequential(
            nn.Linear(bottleneck_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
        )

    def train(self, mode: bool = True):  # noqa: A003
        super().train(mode)
        self.system_encoder.eval()
        return self

    @torch.no_grad()
    def _encode_system_tokens(
        self, texts: list[str], device: torch.device
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if not texts:
            return (
                torch.zeros(0, 1, self.system_encoder.config.hidden_size, device=device),
                torch.zeros(0, 1, dtype=torch.bool, device=device),
            )
        # Head+tail before HF truncation so long assistant drafts keep both lead-in and ending
        # (default tokenizer truncation would keep only the start after the passage prefix).
        truncated = [
            truncate_text_head_tail(
                t,
                self.system_tokenizer,
                max_length=self.max_length,
                head_tokens=128,
                tail_tokens=128,
                prefix=self.passage_prefix,
            )
            for t in texts
        ]
        encoded = self.system_tokenizer(
            [f"{self.passage_prefix}{t}" for t in truncated],
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        ).to(device)
        out = self.system_encoder(**encoded, output_hidden_states=False)
        tokens = out.last_hidden_state
        mask = encoded["attention_mask"].bool()
        return tokens, mask

    @torch.no_grad()
    def embed_texts(
        self, texts: list[str], *, device: torch.device | None = None
    ) -> torch.Tensor:
        """Frozen e5 embeddings for semantic command-overlap supervision."""

        device = device or next(self.parameters()).device
        if not texts:
            hidden = int(self.system_encoder.config.hidden_size)
            return torch.zeros(0, hidden, device=device)
        tokens, mask = self._encode_system_tokens(texts, device)
        return torch.nn.functional.normalize(mean_pool(tokens, mask), dim=-1)

    def _encode_user_queries(
        self, texts: list[str], device: torch.device
    ) -> torch.Tensor:
        encoded = self.user_tokenizer(
            [f"{self.query_prefix}{t}" for t in texts],
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        ).to(device)
        out = self.user_encoder(**encoded, output_hidden_states=True)
        # Use all hidden states including embeddings.
        layer_states = out.hidden_states
        pooled_layers = self.layer_pool(layer_states)
        return mean_pool(pooled_layers, encoded["attention_mask"])

    def encode_sessions(
        self,
        sessions: list[list[dict[str, str]]],
        *,
        device: torch.device | None = None,
    ) -> dict[str, torch.Tensor]:
        """Encode a list of role/content turn lists into bottleneck embeddings."""

        device = device or next(self.parameters()).device
        batch = len(sessions)
        # Collect valid user turns with optional preceding assistant text.
        user_texts: list[str] = []
        system_texts: list[str] = []
        has_system_flags: list[bool] = []
        locations: list[tuple[int, int]] = []
        max_turns = 0
        for b, turns in enumerate(sessions):
            prev_system = ""
            turn_i = 0
            for turn in turns:
                role = turn.get("role", "")
                content = turn.get("content", "")
                if role in {"assistant", "system"}:
                    prev_system = content
                    continue
                if role != "user" or not content.strip():
                    continue
                if turn_i >= self.max_turns:
                    break
                user_texts.append(content)
                system_texts.append(prev_system)
                has_system_flags.append(bool(prev_system.strip()))
                locations.append((b, turn_i))
                turn_i += 1
            max_turns = max(max_turns, turn_i)

        hidden = self.user_encoder.config.hidden_size
        if not user_texts:
            zeros = torch.zeros(batch, self.bottleneck_dim, device=device)
            return {
                "bottleneck": zeros,
                "session": zeros,
                "vicreg": self.vicreg_proj(zeros),
                "turn_mask": torch.ones(batch, 1, dtype=torch.bool, device=device),
            }

        user_q = self._encode_user_queries(user_texts, device)
        # Encode systems that exist; use dummy for empty.
        nonempty_idx = [i for i, flag in enumerate(has_system_flags) if flag]
        system_tokens = torch.zeros(
            len(user_texts), 1, hidden, device=device, dtype=user_q.dtype
        )
        system_mask = torch.zeros(len(user_texts), 1, dtype=torch.bool, device=device)
        if nonempty_idx:
            toks, mask = self._encode_system_tokens(
                [system_texts[i] for i in nonempty_idx], device
            )
            # Pad to common length across all rows.
            max_s = toks.size(1)
            system_tokens = torch.zeros(
                len(user_texts), max_s, hidden, device=device, dtype=user_q.dtype
            )
            system_mask = torch.zeros(len(user_texts), max_s, dtype=torch.bool, device=device)
            system_tokens[nonempty_idx] = toks.to(user_q.dtype)
            system_mask[nonempty_idx] = mask

        has_system = torch.tensor(has_system_flags, device=device)
        fused = self.system_fusion(user_q, system_tokens, system_mask, has_system)

        turn_embeds = torch.zeros(batch, max(1, max_turns), hidden, device=device, dtype=fused.dtype)
        turn_mask = torch.ones(batch, max(1, max_turns), dtype=torch.bool, device=device)
        for (b, t), vec in zip(locations, fused, strict=True):
            turn_embeds[b, t] = vec
            turn_mask[b, t] = False

        session = self.session_encoder(turn_embeds, turn_mask)
        bottleneck = self.bottleneck(session)
        return {
            "bottleneck": bottleneck,
            "session": session,
            "vicreg": self.vicreg_proj(bottleneck),
            "turn_mask": turn_mask,
        }

    def forward(self, sessions: list[list[dict[str, str]]]) -> dict[str, torch.Tensor]:
        return self.encode_sessions(sessions)

    def config_dict(self) -> dict[str, Any]:
        return {
            "user_model_name": self.user_model_name,
            "system_model_name": self.system_model_name,
            "bottleneck_dim": self.bottleneck_dim,
            "session_dim": self.session_encoder.session_dim,
            # Without these, reload rebuilds the session transformer at the build defaults
            # (4 layers / 8 heads) and silently drops the checkpoint's extra layers.
            "session_layers": len(self.session_encoder.transformer.layers),
            "session_heads": self.system_fusion.cross_attn.num_heads,
            "max_turns": self.max_turns,
            "max_length": self.max_length,
            "query_prefix": self.query_prefix,
            "passage_prefix": self.passage_prefix,
        }
