"""Persona-manual decoder with Flamingo-style gated cross-attention (inference)."""

from __future__ import annotations

from typing import Any

import torch
from torch import nn
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

from cue_hf.decoder.flamingo import GatedXAttnDense

DEFAULT_DECODER = "Qwen/Qwen3-0.6B-Base"


class PersonaMemory(nn.Module):
    """Expand bottleneck to a fixed set of persona memory tokens."""

    def __init__(self, bottleneck_dim: int, hidden_dim: int, num_tokens: int = 16) -> None:
        super().__init__()
        self.num_tokens = num_tokens
        self.proj = nn.Sequential(
            nn.Linear(bottleneck_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, num_tokens * hidden_dim),
        )

    def forward(self, bottleneck: torch.Tensor) -> torch.Tensor:
        out = self.proj(bottleneck)
        return out.view(bottleneck.size(0), self.num_tokens, -1)


class CueDecoder(nn.Module):
    def __init__(
        self,
        *,
        lm: nn.Module,
        tokenizer: Any,
        memory: PersonaMemory,
        gated_blocks: nn.ModuleList,
        slot_embeddings: nn.Embedding | None,
        slot_projection: nn.Module | None,
        general_command_slots: int,
        user_specific_command_slots: int,
        style_command_slots: int,
        insert_every: int,
        config: dict[str, Any],
    ) -> None:
        super().__init__()
        self.lm = lm
        self.tokenizer = tokenizer
        self.memory = memory
        self.gated_blocks = gated_blocks
        self.slot_embeddings = slot_embeddings
        self.slot_projection = slot_projection
        self.general_command_slots = int(general_command_slots)
        self.user_specific_command_slots = int(user_specific_command_slots)
        self.style_command_slots = int(style_command_slots)
        self.insert_every = insert_every
        self.config = config
        self._persona_cache: torch.Tensor | None = None
        self._register_hooks()

    def _register_hooks(self) -> None:
        """Wrap selected decoder layers with gated XAttn-Dense."""

        layers = self._get_layers()
        self._hook_handles = []
        block_idx = 0
        for i, layer in enumerate(layers):
            if (i + 1) % self.insert_every != 0:
                continue
            if block_idx >= len(self.gated_blocks):
                break
            gated = self.gated_blocks[block_idx]
            block_idx += 1

            def _make_hook(block: GatedXAttnDense):
                def hook(_module, _inputs, output):
                    persona = self._persona_cache
                    if persona is None:
                        return output
                    if isinstance(output, tuple):
                        hidden = output[0]
                        new_hidden = block(hidden, persona)
                        return (new_hidden,) + output[1:]
                    return block(output, persona)

                return hook

            self._hook_handles.append(layer.register_forward_hook(_make_hook(gated)))

    def _get_layers(self) -> nn.ModuleList:
        if hasattr(self.lm, "model") and hasattr(self.lm.model, "layers"):
            return self.lm.model.layers
        if hasattr(self.lm, "transformer") and hasattr(self.lm.transformer, "h"):
            return self.lm.transformer.h
        raise ValueError("Unsupported decoder architecture: cannot find layers")

    def command_slot_id(self, head: str, slot: int) -> int:
        if head == "general":
            if not 0 <= slot < self.general_command_slots:
                raise ValueError(f"general slot {slot} out of range")
            return slot
        if head == "specific":
            if not 0 <= slot < self.user_specific_command_slots:
                raise ValueError(f"user-specific slot {slot} out of range")
            return self.general_command_slots + slot
        if head == "style":
            if not 0 <= slot < self.style_command_slots:
                raise ValueError(f"style slot {slot} out of range")
            return self.general_command_slots + self.user_specific_command_slots + slot
        raise ValueError(f"unknown command head {head!r}")

    def set_persona(
        self,
        bottleneck: torch.Tensor,
        slot_ids: torch.Tensor | None = None,
    ) -> torch.Tensor:
        # Memory stays fp32; cast encoder bottleneck in for a stable projection.
        dtype = next(self.memory.parameters()).dtype
        persona = self.memory(bottleneck.to(dtype=dtype))
        if slot_ids is not None:
            if self.slot_embeddings is None or self.slot_projection is None:
                raise ValueError("slot_ids provided but decoder has no slot embeddings")
            slot = self.slot_projection(self.slot_embeddings(slot_ids).to(dtype=dtype))
            persona = torch.cat([slot.unsqueeze(1), persona], dim=1)
        self._persona_cache = persona
        return self._persona_cache

    def clear_persona(self) -> None:
        self._persona_cache = None

    @torch.no_grad()
    def generate(
        self,
        *,
        bottleneck: torch.Tensor,
        sentinel_ids: torch.Tensor,
        slot_ids: torch.Tensor | None = None,
        max_new_tokens: int = 1024,
        temperature: float = 0.0,
        top_p: float = 1.0,
    ) -> list[str]:
        self.set_persona(bottleneck, slot_ids=slot_ids)
        embed = self.lm.get_input_embeddings()
        batch = bottleneck.size(0)
        sentinel = embed(sentinel_ids).unsqueeze(0).expand(batch, -1, -1)
        attn = torch.ones(batch, sentinel.size(1), device=bottleneck.device, dtype=torch.long)
        generated = self.lm.generate(
            inputs_embeds=sentinel,
            attention_mask=attn,
            max_new_tokens=max_new_tokens,
            do_sample=temperature > 0,
            temperature=temperature if temperature > 0 else None,
            top_p=top_p if temperature > 0 else None,
            pad_token_id=(
                self.tokenizer.pad_token_id if self.tokenizer.pad_token_id is not None else self.tokenizer.eos_token_id
            ),
            eos_token_id=self.tokenizer.eos_token_id,
        )
        return [self.tokenizer.decode(seq, skip_special_tokens=True).strip() for seq in generated]

    @classmethod
    def build(
        cls,
        *,
        model_name: str = DEFAULT_DECODER,
        bottleneck_dim: int = 1024,
        num_persona_tokens: int = 16,
        insert_every: int = 1,
        num_heads: int = 8,
        general_command_slots: int = 0,
        user_specific_command_slots: int = 0,
        style_command_slots: int = 0,
        slot_embedding_dim: int = 64,
        dtype: torch.dtype = torch.bfloat16,
        pretrained_lm: bool = True,
    ) -> CueDecoder:
        tokenizer = AutoTokenizer.from_pretrained(
            model_name, trust_remote_code=True, clean_up_tokenization_spaces=False
        )
        tokenizer.padding_side = "right"
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        if pretrained_lm:
            # low_cpu_mem_usage=False forces real weights on CPU (no accelerate meta init).
            # With meta init the tied lm_head is an empty meta tensor and a later .to(device)
            # fails with "Cannot copy out of meta tensor".
            try:
                lm = AutoModelForCausalLM.from_pretrained(
                    model_name, dtype=dtype, trust_remote_code=True, low_cpu_mem_usage=False
                )
            except TypeError:
                lm = AutoModelForCausalLM.from_pretrained(
                    model_name,
                    torch_dtype=dtype,
                    trust_remote_code=True,
                    low_cpu_mem_usage=False,
                )
        else:
            # A complete CUE checkpoint replaces every LM weight, so only the architecture
            # config is needed. Skips a redundant base-weight download.
            lm_config = AutoConfig.from_pretrained(model_name, trust_remote_code=True)
            lm = AutoModelForCausalLM.from_config(lm_config, trust_remote_code=True).to(dtype=dtype)
        # Ensure tied weights (lm_head <-> embed_tokens) are materialized, not left on meta.
        if hasattr(lm, "tie_weights"):
            lm.tie_weights()
        lm.generation_config.pad_token_id = tokenizer.pad_token_id
        hidden = lm.get_input_embeddings().embedding_dim
        layers = lm.model.layers if hasattr(lm, "model") and hasattr(lm.model, "layers") else lm.transformer.h
        n_blocks = max(1, len(layers) // max(1, insert_every))
        # Keep conditioning modules in fp32; GatedXAttnDense casts activations internally.
        gated = nn.ModuleList([GatedXAttnDense(hidden, num_heads=num_heads) for _ in range(n_blocks)])
        memory = PersonaMemory(bottleneck_dim, hidden, num_persona_tokens)
        num_slots = (
            max(0, int(general_command_slots))
            + max(0, int(user_specific_command_slots))
            + max(0, int(style_command_slots))
        )
        slot_embeddings = nn.Embedding(num_slots, max(1, int(slot_embedding_dim))) if num_slots else None
        slot_projection = (
            nn.Sequential(
                nn.Linear(max(1, int(slot_embedding_dim)), hidden),
                nn.LayerNorm(hidden),
            )
            if num_slots
            else None
        )
        config = {
            "model_name": model_name,
            "dtype": str(dtype).removeprefix("torch."),
            "bottleneck_dim": bottleneck_dim,
            "num_persona_tokens": num_persona_tokens,
            "insert_every": insert_every,
            "num_heads": num_heads,
            "general_command_slots": int(general_command_slots),
            "user_specific_command_slots": int(user_specific_command_slots),
            "style_command_slots": int(style_command_slots),
            "slot_embedding_dim": int(slot_embedding_dim),
            "injection": "flamingo_gated_xattn_dense",
        }
        return cls(
            lm=lm,
            tokenizer=tokenizer,
            memory=memory,
            gated_blocks=gated,
            slot_embeddings=slot_embeddings,
            slot_projection=slot_projection,
            general_command_slots=general_command_slots,
            user_specific_command_slots=user_specific_command_slots,
            style_command_slots=style_command_slots,
            insert_every=insert_every,
            config=config,
        )

    def gate_magnitudes(self) -> dict[str, float]:
        xs = [float(torch.tanh(b.alpha_xattn).detach().cpu()) for b in self.gated_blocks]
        ds = [float(torch.tanh(b.alpha_dense).detach().cpu()) for b in self.gated_blocks]
        return {
            "xattn_gate_mean": sum(xs) / max(1, len(xs)),
            "dense_gate_mean": sum(ds) / max(1, len(ds)),
            "xattn_gate_max": max(xs) if xs else 0.0,
            "dense_gate_max": max(ds) if ds else 0.0,
        }
