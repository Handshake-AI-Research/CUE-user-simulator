"""CueModel: encode real sessions or sample synthetic users, then decode manuals."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import torch
from transformers import PreTrainedModel

from cue_hf.checkpoint import (
    LAYOUT_JOINT,
    LAYOUT_REFINED,
    config_from_joint_dir,
    detect_layout,
    load_joint_state,
    resolve_local_dir,
    resolve_refined,
)
from cue_hf.configuration_cue import CueConfig
from cue_hf.decoder.model import CueDecoder
from cue_hf.encoder.model import CueEncoder
from cue_hf.example_pool import META_NAME, POOL_DIRNAME, ExamplePool, inject_examples
from cue_hf.log import log, warn
from cue_hf.sampler.sample import SAMPLER_NAME, CueSampler, load_sampler
from cue_hf.schema import (
    COMMAND_BLOCK_SENTINEL,
    COMMAND_SLOT_SENTINEL,
    GENERAL_COMMAND_SENTINEL,
    STYLE_COMMAND_SENTINEL,
    USER_SPECIFIC_SENTINEL,
    clean_generated_command,
    merge_dual_manual_blocks,
    parse_manual,
)
from cue_hf.session_preprocess import preprocess_turns

TAG = "cue"
_WEIGHT_NAMES = ("model.safetensors", "pytorch_model.bin", "model.pt")


def command_token_jaccard(a: str, b: str) -> float:
    """Token Jaccard similarity on casefolded whitespace splits."""

    ta = set(a.casefold().split())
    tb = set(b.casefold().split())
    if not ta and not tb:
        return 1.0
    return len(ta & tb) / max(1, len(ta | tb))


def select_command_threshold(
    candidates: list[str],
    kept: list[str],
    *,
    near_dup_jaccard: float = 0.8,
) -> str | None:
    """Pick the first candidate that is not a near-duplicate of already-kept commands.

    Decorrelates paraphrases without repulsive MMR (which pushed toward semantic
    opposites). Sample order is treated as relevance (earlier ≈ closer to the mode).
    """

    cleaned = [str(c).strip() for c in candidates if str(c).strip()]
    if not cleaned:
        return None
    thresh = min(1.0, max(0.0, float(near_dup_jaccard)))
    for command in cleaned:
        max_sim = max(command_token_jaccard(command, prev) for prev in kept) if kept else 0.0
        if kept and max_sim >= thresh:
            continue
        return command
    return None


class CueModel(PreTrainedModel):
    """A CUE user model: session encoder + persona-manual decoder.

    Two ways in, both producing a CUE bottleneck that the decoder turns into a manual:

    - real user: ``encode(sessions)`` / ``generate_manual(sessions=...)``
    - sampled user: ``sample(n=...)`` / ``sample_user(n=...)`` (needs sampler weights)

    Sessions are turn lists, not ``input_ids``; the per-turn tokenizers live inside the
    encoder and the decoder. See ``CueProcessor`` for chat-format normalization.
    """

    config_class = CueConfig
    base_model_prefix = "cue"
    supports_gradient_checkpointing = False

    def __init__(self, config: CueConfig, *, pretrained_backbones: bool = True) -> None:
        super().__init__(config)
        enc_cfg = config.encoder
        dec_cfg = config.decoder
        self.encoder = CueEncoder(
            user_model_name=str(enc_cfg["user_model_name"]),
            system_model_name=str(enc_cfg["system_model_name"]),
            bottleneck_dim=int(enc_cfg["bottleneck_dim"]),
            session_dim=int(enc_cfg["session_dim"]),
            session_layers=int(enc_cfg["session_layers"]),
            session_heads=int(enc_cfg["session_heads"]),
            max_turns=int(enc_cfg["max_turns"]),
            max_length=int(enc_cfg["max_length"]),
            query_prefix=str(enc_cfg["query_prefix"]),
            passage_prefix=str(enc_cfg["passage_prefix"]),
            pretrained_backbones=pretrained_backbones,
        )
        use_slots = bool(config.command_slot_decode)
        dtype = getattr(torch, str(dec_cfg.get("dtype", "bfloat16")), torch.bfloat16)
        self.decoder = CueDecoder.build(
            model_name=str(dec_cfg["model_name"]),
            bottleneck_dim=int(enc_cfg["bottleneck_dim"]),
            num_persona_tokens=int(dec_cfg["num_persona_tokens"]),
            insert_every=int(dec_cfg["insert_every"]),
            num_heads=int(dec_cfg["num_heads"]),
            general_command_slots=config.general_command_slots if use_slots else 0,
            user_specific_command_slots=(config.user_specific_command_slots if use_slots else 0),
            style_command_slots=config.style_command_slots if use_slots else 0,
            slot_embedding_dim=config.slot_embedding_dim,
            dtype=dtype,
            pretrained_lm=pretrained_backbones,
        )
        self._sampler: CueSampler | None = None
        self._example_pool: ExamplePool | None = None

    # --- tokenizers (per turn / per manual; there is no session-level tokenizer) ---

    @property
    def encoder_tokenizer(self) -> Any:
        """Tokenizer for user turns. Applied per turn inside ``encode``."""

        return self.encoder.user_tokenizer

    @property
    def decoder_tokenizer(self) -> Any:
        """Tokenizer of the manual LM, used for slot decode and manual parsing."""

        return self.decoder.tokenizer

    # --- attaching optional side artifacts ---

    @property
    def sampler(self) -> CueSampler | None:
        return self._sampler

    @property
    def example_pool(self) -> ExamplePool | None:
        return self._example_pool

    def attach_sampler(self, checkpoint: str | Path, **kwargs: Any) -> CueSampler:
        """Load a diffusion prior (``sampler.pt``) from a directory or Hub repo."""

        source = Path(checkpoint)
        if not source.is_dir():
            from huggingface_hub import hf_hub_download

            # Refined repos point sampler_id at the joint repo, so snapshotting it would
            # pull that model's weights too: fetch the one file instead.
            source = Path(hf_hub_download(str(checkpoint), SAMPLER_NAME, **kwargs))
        self._sampler = load_sampler(source, device=str(self.device))
        return self._sampler

    def attach_example_pool(self, source: str | Path, *, split: str = "train") -> ExamplePool:
        """Load an example pool from a cue_training pool dir, or a dataset (Hub id or path)."""

        if _is_cue_pool_dir(source):
            self._example_pool = ExamplePool.from_directory(source)
        else:
            self._example_pool = ExamplePool.from_hub_dataset(str(source), split=split)
        if self._example_pool.dim != self.config.bottleneck_dim:
            raise ValueError(
                f"example pool dim {self._example_pool.dim} does not match model "
                f"bottleneck {self.config.bottleneck_dim}"
            )
        return self._example_pool

    # --- encode ---

    @torch.no_grad()
    def encode(
        self,
        sessions: list[list[dict[str, str]]],
        *,
        session_preprocess: str | None = None,
    ) -> torch.Tensor:
        """Encode turn lists into raw CUE bottleneck vectors [B, D] (not normalized)."""

        mode = session_preprocess or self.config.session_preprocess
        prepared = [preprocess_turns(list(turns), mode) for turns in sessions]
        return self.encoder.encode_sessions(prepared)["bottleneck"]

    # --- sample ---

    def _require_sampler(self) -> CueSampler:
        if self._sampler is None:
            raise RuntimeError(
                "no sampler weights loaded. Pass sampler_checkpoint=... to "
                "from_pretrained, or call attach_sampler('path/or/repo'). Sampler "
                "training lives in cue_training (`cue-train sampler-train`)."
            )
        return self._sampler

    @torch.no_grad()
    def sample(
        self,
        *,
        n: int = 1,
        condition: np.ndarray | torch.Tensor | list | None = None,
        steps: int | None = None,
        guidance_w: float | None = None,
        project_manifold: bool | None = None,
        seed: int = 0,
    ) -> torch.Tensor:
        """Sample synthetic CUE embeddings [n, D] from the diffusion prior."""

        return self._require_sampler().sample(
            n=n,
            condition=condition,
            steps=steps,
            guidance_w=guidance_w,
            project_manifold=project_manifold,
            seed=seed,
        )

    @torch.no_grad()
    def sample_user(
        self,
        *,
        n: int = 1,
        condition: np.ndarray | torch.Tensor | list | None = None,
        steps: int | None = None,
        guidance_w: float | None = None,
        project_manifold: bool | None = None,
        seed: int = 0,
        **generate_kwargs: Any,
    ) -> dict[str, Any]:
        """Sample synthetic users and decode their manuals in one call."""

        embeddings = self.sample(
            n=n,
            condition=condition,
            steps=steps,
            guidance_w=guidance_w,
            project_manifold=project_manifold,
            seed=seed,
        )
        manuals = self.generate_manual(embeddings=embeddings, **generate_kwargs)
        return {"embeddings": embeddings, "manuals": manuals}

    # --- generate manuals ---

    @torch.no_grad()
    def generate_manual(
        self,
        *,
        sessions: list[list[dict[str, str]]] | None = None,
        embeddings: torch.Tensor | np.ndarray | None = None,
        max_new_tokens: int | None = None,
        temperature: float | None = None,
        example_retrieval: bool = False,
        k_sessions: int | None = None,
        n_general: int | None = None,
        n_specific: int | None = None,
        exclude_session_ids: list[str | None] | None = None,
        session_preprocess: str | None = None,
    ) -> list[dict[str, Any] | None]:
        """Decode a persona manual per session (or per sampled embedding)."""

        if (sessions is None) == (embeddings is None):
            raise ValueError("pass exactly one of sessions= (real users) or embeddings= (sampled users)")
        if sessions is not None:
            bottleneck = self.encode(sessions, session_preprocess=session_preprocess)
        else:
            bottleneck = torch.as_tensor(embeddings, dtype=torch.float32)
            if bottleneck.ndim == 1:
                bottleneck = bottleneck.unsqueeze(0)
            bottleneck = bottleneck.to(self.device)
        manuals = self._manuals_from_bottleneck(
            bottleneck,
            max_new_tokens=(self.config.decode_max_new_tokens if max_new_tokens is None else int(max_new_tokens)),
            temperature=(self.config.decode_temperature if temperature is None else float(temperature)),
        )
        if not example_retrieval:
            return manuals
        return self._inject_examples(
            manuals,
            bottleneck,
            k_sessions=k_sessions,
            n_general=n_general,
            n_specific=n_specific,
            exclude_session_ids=exclude_session_ids,
        )

    def _inject_examples(
        self,
        manuals: list[dict[str, Any] | None],
        bottleneck: torch.Tensor,
        *,
        k_sessions: int | None,
        n_general: int | None,
        n_specific: int | None,
        exclude_session_ids: list[str | None] | None,
    ) -> list[dict[str, Any] | None]:
        if self._example_pool is None and self.config.example_pool_id:
            self.attach_example_pool(self.config.example_pool_id)
        if self._example_pool is None:
            warn(
                TAG,
                "example_retrieval=True but no example pool is loaded; returning manuals "
                "without retrieved examples. Set config.example_pool_id or call "
                "attach_example_pool(...). Build a pool with cue_training's "
                "`cue-train build-example-pool`.",
            )
            return manuals
        if not self.config.dual_decode:
            warn(TAG, "example retrieval is a no-op for single-head (non-dual) decode")
            return manuals
        vectors = bottleneck.detach().float().cpu().numpy()
        excludes = exclude_session_ids or [None] * len(manuals)
        return [
            inject_examples(
                manual,
                pool=self._example_pool,
                cue_embedding=vector,
                k_sessions=(self.config.example_retrieval_k_sessions if k_sessions is None else int(k_sessions)),
                n_general=(self.config.example_retrieval_n_general if n_general is None else int(n_general)),
                n_specific=(self.config.example_retrieval_n_specific if n_specific is None else int(n_specific)),
                exclude_session_id=exclude,
            )
            for manual, vector, exclude in zip(manuals, vectors, excludes, strict=True)
        ]

    def _sentinel_ids(self, sentinel: str) -> torch.Tensor:
        return self.decoder.tokenizer(sentinel, add_special_tokens=False, return_tensors="pt")["input_ids"][0].to(
            self.device
        )

    @torch.no_grad()
    def _manuals_from_bottleneck(
        self,
        bottleneck: torch.Tensor,
        *,
        max_new_tokens: int,
        temperature: float,
    ) -> list[dict[str, Any] | None]:
        if not self.config.dual_decode:
            texts = self.decoder.generate(
                bottleneck=bottleneck,
                sentinel_ids=self._sentinel_ids(COMMAND_BLOCK_SENTINEL),
                max_new_tokens=max_new_tokens,
                temperature=temperature,
            )
            return [parse_manual(text) for text in texts]

        if self.config.command_slot_decode:
            blocks = {
                head: self.generate_command_slot_blocks(
                    bottleneck,
                    head=head,
                    num_slots=num_slots,
                    max_new_tokens=max_new_tokens,
                    temperature=temperature,
                )
                if num_slots > 0
                else [{"commands": [], "examples": []}] * bottleneck.size(0)
                for head, num_slots in (
                    ("general", self.config.general_command_slots),
                    ("specific", self.config.user_specific_command_slots),
                    ("style", self.config.style_command_slots),
                )
            }
            merged = [
                merge_dual_manual_blocks(general, specific, style)
                for general, specific, style in zip(blocks["general"], blocks["specific"], blocks["style"], strict=True)
            ]
            return [manual if manual.get("commands") else None for manual in merged]

        empty = {"commands": [], "examples": []}
        general_texts = self.decoder.generate(
            bottleneck=bottleneck,
            sentinel_ids=self._sentinel_ids(GENERAL_COMMAND_SENTINEL),
            max_new_tokens=max_new_tokens,
            temperature=temperature,
        )
        specific_texts = self.decoder.generate(
            bottleneck=bottleneck,
            sentinel_ids=self._sentinel_ids(USER_SPECIFIC_SENTINEL),
            max_new_tokens=max_new_tokens,
            temperature=temperature,
        )
        style_texts = None
        if self.config.style_command_slots > 0:
            style_texts = self.decoder.generate(
                bottleneck=bottleneck,
                sentinel_ids=self._sentinel_ids(STYLE_COMMAND_SENTINEL),
                max_new_tokens=max_new_tokens,
                temperature=temperature,
            )
        manuals: list[dict[str, Any] | None] = []
        for index, (general_raw, specific_raw) in enumerate(zip(general_texts, specific_texts, strict=True)):
            general = parse_manual(general_raw) or dict(empty)
            specific = parse_manual(specific_raw) or dict(empty)
            style = (parse_manual(style_texts[index]) or dict(empty)) if style_texts is not None else dict(empty)
            merged_manual = merge_dual_manual_blocks(general, specific, style)
            manuals.append(merged_manual if merged_manual.get("commands") else None)
        return manuals

    @torch.no_grad()
    def generate_command_slot_blocks(
        self,
        bottleneck: torch.Tensor,
        *,
        head: str,
        num_slots: int,
        max_new_tokens: int = 128,
        temperature: float = 0.0,
        top_p: float = 1.0,
    ) -> list[dict[str, Any]]:
        """Decode one short command per slot, dropping near-duplicates.

        Greedy decode is deterministic, so a slot whose mode is ``<NO_COMMAND>`` stays
        empty forever and the manual comes back nearly blank. ``decode_noop_retries``
        resamples just those rows at ``decode_noop_temperature`` instead of dropping them.
        """

        per_sample: list[list[str]] = [[] for _ in range(bottleneck.size(0))]
        sentinel = self._sentinel_ids(COMMAND_SLOT_SENTINEL)
        near_dup = self.config.slot_dedup_jaccard
        for slot in range(num_slots):
            slot_ids = torch.full(
                (bottleneck.size(0),),
                self.decoder.command_slot_id(head, slot),
                device=bottleneck.device,
                dtype=torch.long,
            )
            texts = self.decoder.generate(
                bottleneck=bottleneck,
                sentinel_ids=sentinel,
                slot_ids=slot_ids,
                max_new_tokens=min(max_new_tokens, 128),
                temperature=temperature,
                top_p=top_p,
            )
            pending: list[int] = []
            for row, (commands, raw) in enumerate(zip(per_sample, texts, strict=True)):
                chosen = select_command_threshold([clean_generated_command(raw)], commands, near_dup_jaccard=near_dup)
                if chosen:
                    commands.append(chosen)
                else:
                    pending.append(row)
            for attempt in range(max(0, self.config.decode_noop_retries)):
                if not pending:
                    break
                index = torch.tensor(pending, device=bottleneck.device, dtype=torch.long)
                retry = self.decoder.generate(
                    bottleneck=bottleneck[index],
                    sentinel_ids=sentinel,
                    slot_ids=slot_ids[index],
                    max_new_tokens=min(max_new_tokens, 128),
                    temperature=max(1e-3, self.config.decode_noop_temperature),
                    top_p=top_p,
                )
                still: list[int] = []
                for row, raw in zip(pending, retry, strict=True):
                    chosen = select_command_threshold(
                        [clean_generated_command(raw)],
                        per_sample[row],
                        near_dup_jaccard=near_dup,
                    )
                    if chosen:
                        per_sample[row].append(chosen)
                    else:
                        still.append(row)
                pending = still
                _ = attempt
        return [{"commands": commands, "examples": []} for commands in per_sample]

    # --- HF surface ---

    def forward(
        self,
        sessions: list[list[dict[str, str]]] | None = None,
        input_ids: torch.Tensor | None = None,
        **kwargs: Any,
    ) -> torch.Tensor:
        """Encode sessions. CUE has no single-sequence ``input_ids`` entrypoint."""

        if input_ids is not None or sessions is None:
            raise TypeError(
                "CueModel encodes dialogue turns, not a single token sequence. Pass "
                "sessions=[[{'role': 'user', 'content': ...}, ...]] to forward/encode, "
                "or use CueProcessor to normalize chat messages. To decode a manual use "
                "generate_manual(sessions=...) or sample_user(n=...)."
            )
        return self.encode(sessions, **kwargs)

    @classmethod
    def from_pretrained(  # type: ignore[override]
        cls,
        pretrained_model_name_or_path: str | Path,
        *,
        device: str | torch.device = "cpu",
        sampler_checkpoint: str | Path | None = None,
        example_pool: str | Path | None = None,
        revision: str | None = None,
        token: str | bool | None = None,
        **kwargs: Any,
    ) -> CueModel:
        """Load a cue-hf repo, a joint training checkpoint, or a refinement checkpoint."""

        directory = resolve_local_dir(pretrained_model_name_or_path, revision=revision, token=token)
        layout = detect_layout(directory)
        if layout == LAYOUT_REFINED:
            joint_dir, decoder_state = resolve_refined(directory)
            model = cls.from_pretrained(joint_dir, device=device, revision=revision, token=token, **kwargs)
            model.decoder.load_state_dict(torch.load(decoder_state, map_location=str(device), weights_only=False))
            model._retie_lm()
            log(TAG, f"loaded refined decoder from {decoder_state}")
        else:
            if layout == LAYOUT_JOINT:
                config = config_from_joint_dir(directory)
                state = load_joint_state(directory, map_location="cpu")
            else:
                config = CueConfig.from_pretrained(directory, **kwargs)
                state = _load_native_state(directory)
            # pretrained_backbones=False: every backbone weight is about to be replaced by
            # the checkpoint, so only the base architecture configs are needed.
            model = cls(config, pretrained_backbones=False)
            model._load_cue_state(state)
        model.eval()
        model.to(device)
        sampler_source = sampler_checkpoint or model.config.sampler_id
        if sampler_source:
            model.attach_sampler(sampler_source, revision=revision, token=token)
        elif (directory / "sampler.pt").is_file():
            model.attach_sampler(directory)
        if _is_cue_pool_dir(directory):
            model.attach_example_pool(directory)
        elif example_pool:
            model.attach_example_pool(example_pool)
        return model

    def _retie_lm(self) -> None:
        # Qwen3 ties lm_head<->embed_tokens, so checkpoints omit lm_head.weight. Building
        # from config and loading a partial state dict can leave lm_head untied (which
        # decodes as gibberish), so re-tie after every load.
        lm = getattr(self.decoder, "lm", None)
        if lm is not None and hasattr(lm, "tie_weights"):
            lm.tie_weights()

    def _load_cue_state(self, state: dict[str, Any]) -> None:
        incompatible = self.load_state_dict(state, strict=False)
        self._retie_lm()
        missing = [k for k in getattr(incompatible, "missing_keys", []) if "lm_head" not in k]
        if missing:
            warn(TAG, f"{len(missing)} missing checkpoint keys (e.g. {missing[:3]})")

    def save_pretrained(self, save_directory: str | Path, **kwargs: Any) -> None:
        """Save config + full weights (trained backbones and LM included)."""

        try:
            super().save_pretrained(save_directory, **kwargs)
        except RuntimeError as exc:
            if "share memory" not in str(exc) and "shared tensors" not in str(exc):
                raise
            # Tied LM embeddings can defeat safetensors' no-aliasing rule.
            warn(TAG, f"safetensors rejected shared tensors ({exc}); saving torch weights")
            super().save_pretrained(save_directory, **{**kwargs, "safe_serialization": False})


def _is_cue_pool_dir(source: str | Path) -> bool:
    """True for a cue_training pool directory (or a checkpoint dir containing one)."""

    path = Path(source)
    return (path / META_NAME).is_file() or (path / POOL_DIRNAME / META_NAME).is_file()


def _load_native_state(directory: Path) -> dict[str, Any]:
    for name in _WEIGHT_NAMES:
        path = directory / name
        if not path.is_file():
            continue
        if path.suffix == ".safetensors":
            from safetensors.torch import load_file

            return load_file(str(path))
        payload = torch.load(path, map_location="cpu", weights_only=False)
        return payload.get("model", payload) if isinstance(payload, dict) else payload
    raise FileNotFoundError(f"{directory} has none of {_WEIGHT_NAMES}; not a cue-hf model directory")
