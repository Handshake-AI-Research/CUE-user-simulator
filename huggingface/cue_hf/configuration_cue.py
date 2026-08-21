"""Configuration for CUE user models."""

from __future__ import annotations

from typing import Any

from transformers import PretrainedConfig

from cue_hf.decoder.model import DEFAULT_DECODER
from cue_hf.encoder.model import (
    DEFAULT_ENCODER,
    DEFAULT_PASSAGE_PREFIX,
    DEFAULT_QUERY_PREFIX,
)
from cue_hf.session_preprocess import normalize_session_preprocess

# Defaults mirror the final CUE joint-training config (ModernBERT turn encoder).
DEFAULT_ENCODER_CONFIG: dict[str, Any] = {
    "user_model_name": DEFAULT_ENCODER,
    "system_model_name": DEFAULT_ENCODER,
    "bottleneck_dim": 1024,
    "session_dim": 1024,
    "session_layers": 4,
    "session_heads": 8,
    "max_turns": 64,
    "max_length": 256,
    "query_prefix": DEFAULT_QUERY_PREFIX,
    "passage_prefix": DEFAULT_PASSAGE_PREFIX,
}

DEFAULT_DECODER_CONFIG: dict[str, Any] = {
    "model_name": DEFAULT_DECODER,
    "num_persona_tokens": 16,
    "insert_every": 4,
    "num_heads": 8,
    "dtype": "bfloat16",
}


class CueConfig(PretrainedConfig):
    """CUE encoder/decoder architecture plus inference-time parity defaults.

    ``encoder`` and ``decoder`` keep the nested shape of cue_training's
    ``unified_config.json`` so exported checkpoints map across without translation.
    Decode defaults follow the final rollout config (greedy, slot dedup 0.8,
    retrieval quotas 8/2/6).
    """

    model_type = "cue"

    def __init__(
        self,
        encoder: dict[str, Any] | None = None,
        decoder: dict[str, Any] | None = None,
        *,
        dual_decode: bool = True,
        command_slot_decode: bool = True,
        general_command_slots: int = 5,
        user_specific_command_slots: int = 5,
        style_command_slots: int = 5,
        slot_embedding_dim: int = 64,
        target_max_tokens: int = 256,
        session_preprocess: str | None = "full",
        decode_temperature: float = 0.0,
        decode_max_new_tokens: int = 2048,
        slot_dedup_jaccard: float = 0.8,
        decode_noop_retries: int = 0,
        decode_noop_temperature: float = 0.7,
        example_pool_id: str | None = None,
        example_retrieval_k_sessions: int = 8,
        example_retrieval_n_general: int = 2,
        example_retrieval_n_specific: int = 6,
        sampler_id: str | None = None,
        **kwargs: Any,
    ) -> None:
        self.encoder = {**DEFAULT_ENCODER_CONFIG, **(encoder or {})}
        self.decoder = {**DEFAULT_DECODER_CONFIG, **(decoder or {})}
        self.dual_decode = bool(dual_decode)
        self.command_slot_decode = bool(command_slot_decode)
        self.general_command_slots = int(general_command_slots)
        self.user_specific_command_slots = int(user_specific_command_slots)
        self.style_command_slots = int(style_command_slots)
        self.slot_embedding_dim = int(slot_embedding_dim)
        self.target_max_tokens = int(target_max_tokens)
        # A config exported without an explicit mode carries null, and str(None) would reach
        # the encoder as the literal "None".
        self.session_preprocess = normalize_session_preprocess(session_preprocess)
        self.decode_temperature = float(decode_temperature)
        self.decode_max_new_tokens = int(decode_max_new_tokens)
        self.slot_dedup_jaccard = float(slot_dedup_jaccard)
        self.decode_noop_retries = int(decode_noop_retries)
        self.decode_noop_temperature = float(decode_noop_temperature)
        self.example_pool_id = example_pool_id
        self.example_retrieval_k_sessions = int(example_retrieval_k_sessions)
        self.example_retrieval_n_general = int(example_retrieval_n_general)
        self.example_retrieval_n_specific = int(example_retrieval_n_specific)
        self.sampler_id = sampler_id
        super().__init__(**kwargs)

    @property
    def bottleneck_dim(self) -> int:
        return int(self.encoder["bottleneck_dim"])

    @classmethod
    def from_cue_config(cls, payload: dict[str, Any]) -> CueConfig:
        """Build from cue_training's ``unified_config.json`` ({"train": ..., "model": ...})."""

        train_cfg = payload.get("train") or {}
        model_cfg = payload.get("model") or {}
        merged: dict[str, Any] = {}
        for key in (
            "dual_decode",
            "command_slot_decode",
            "general_command_slots",
            "user_specific_command_slots",
            "style_command_slots",
            "slot_embedding_dim",
            "target_max_tokens",
        ):
            if key in train_cfg:
                merged[key] = train_cfg[key]
            if key in model_cfg:
                merged[key] = model_cfg[key]
        # The model block records what was actually built and wins; the train block
        # supplies keys the model block omits (notably decoder dtype).
        return cls(
            encoder={**(train_cfg.get("encoder") or {}), **(model_cfg.get("encoder") or {})},
            decoder={**(train_cfg.get("decoder") or {}), **(model_cfg.get("decoder") or {})},
            **merged,
        )
