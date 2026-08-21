"""Load a CUE checkpoint and produce persona manuals."""

from __future__ import annotations

import importlib
from typing import TYPE_CHECKING, Any, Literal

import torch
from transformers import AutoModel

from cue.conversation import normalize_messages
from cue.manual import PersonaManual
from cue.steering import render_steering

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence

DEFAULT_REPO = "handshake-ai-research/cue"
_POOL_UNAVAILABLE = (
    "this model build cannot attach an example pool; install cue-simulator or use a "
    "Hub repo that bundles the full CueModel"
)
_SAMPLER_UNAVAILABLE = (
    "sampling requires CueModel.sample; install cue-simulator or use a Hub repo "
    "bundled with the full remote-code package"
)
_Source = Literal["conditioned", "sampled"]


def _load_model(repo_id: str, *, device: str, token: str | None, **kwargs: Any) -> Any:
    """Prefer ``cue_hf`` when installed; otherwise Hub remote code."""

    try:
        cue_model = importlib.import_module("cue_hf").CueModel
        return cue_model.from_pretrained(repo_id, device=device, token=token, **kwargs)
    except ImportError:
        model = AutoModel.from_pretrained(repo_id, trust_remote_code=True, token=token, **kwargs)
        if device and str(device) != "cpu":
            model = model.to(device)
        return model.eval()


def _commands(block: Mapping[str, Any] | None) -> list[str]:
    if not block:
        return []
    out: list[str] = []
    for entry in block.get("commands") or []:
        if isinstance(entry, dict):
            text = str(entry.get("text") or entry.get("command") or "").strip()
        else:
            text = str(entry).strip()
        if text:
            out.append(text)
    return out


def _examples(block: Mapping[str, Any] | None) -> list[str]:
    if not block:
        return []
    return [str(x).strip() for x in (block.get("examples") or []) if str(x).strip()]


def _manual_from_blocks(
    *,
    general: Mapping[str, Any] | None,
    specific: Mapping[str, Any] | None,
    style: Mapping[str, Any] | None,
    embedding: torch.Tensor | None,
    source: _Source,
) -> PersonaManual:
    g, s, st = _commands(general), _commands(specific), _commands(style)
    examples = _examples(specific) + _examples(style) + _examples(general)
    # Dedup while preserving order.
    seen: set[str] = set()
    unique_ex: list[str] = []
    for item in examples:
        if item not in seen:
            seen.add(item)
            unique_ex.append(item)
    commands = g + s + st
    prompt = render_steering(general=g, specific=s, style=st, examples=unique_ex)
    emb = None
    if embedding is not None:
        emb = embedding.detach().float().cpu().reshape(-1).tolist()
    return PersonaManual(
        commands=commands,
        examples=unique_ex,
        steering_prompt=prompt,
        general=g,
        specific=s,
        style=st,
        embedding=emb,
        source=source,
        raw={"general": general, "specific": specific, "style": style},
    )


def _manual_from_merged(
    manual: Mapping[str, Any] | None,
    *,
    embedding: torch.Tensor | None,
    source: _Source,
) -> PersonaManual:
    commands = _commands(manual)
    examples = _examples(manual)
    prompt = render_steering(commands=commands, examples=examples)
    emb = None
    if embedding is not None:
        emb = embedding.detach().float().cpu().reshape(-1).tolist()
    return PersonaManual(
        commands=commands,
        examples=examples,
        steering_prompt=prompt,
        embedding=emb,
        source=source,
        raw=dict(manual) if manual else None,
    )


class Cue:
    """Application entry point: condition on a conversation, or sample a user."""

    def __init__(self, model: Any, *, default_preprocess: str = "full") -> None:
        self.model = model
        self.default_preprocess = default_preprocess

    @classmethod
    def from_pretrained(
        cls,
        repo_id: str = DEFAULT_REPO,
        *,
        device: str = "cpu",
        token: str | None = None,
        example_pool: str | None = None,
        session_preprocess: str = "full",
        **kwargs: Any,
    ) -> Cue:
        """Load weights from a Hub repo (or local export directory).

        Install ``cue-simulator`` for the full API, or rely on the model repo's bundled
        remote code (``trust_remote_code``) when you only need inference.
        """

        model = _load_model(repo_id, device=device, token=token, **kwargs)
        cue = cls(model, default_preprocess=session_preprocess)
        if example_pool:
            cue.attach_example_pool(example_pool)
        return cue

    def attach_example_pool(self, pool: str) -> None:
        if not hasattr(self.model, "attach_example_pool"):
            raise RuntimeError(_POOL_UNAVAILABLE)
        self.model.attach_example_pool(pool)

    def _sessions(
        self,
        messages: Sequence[Mapping[str, Any]] | Sequence[str] | str,
    ) -> list[list[dict[str, str]]]:
        # Encode applies session_preprocess itself; do not preprocess here.
        return [normalize_messages(messages)]

    @torch.no_grad()
    def from_conversation(
        self,
        messages: Sequence[Mapping[str, Any]] | Sequence[str] | str,
        *,
        session_preprocess: str | None = None,
        example_retrieval: bool = False,
        max_new_tokens: int | None = None,
        temperature: float | None = None,
    ) -> PersonaManual:
        """Encode a dialogue history and decode a user-conditioned persona manual."""

        sessions = self._sessions(messages)
        return self._decode_sessions(
            sessions,
            source="conditioned",
            example_retrieval=example_retrieval,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            session_preprocess=session_preprocess,
        )[0]

    @torch.no_grad()
    def sample(
        self,
        *,
        n: int = 1,
        seed: int = 0,
        condition: Any = None,
        max_new_tokens: int | None = None,
        temperature: float | None = None,
        example_retrieval: bool = False,
        **sample_kwargs: Any,
    ) -> PersonaManual | list[PersonaManual]:
        """Draw synthetic users from the diffusion prior (needs ``sampler.pt``)."""

        if not hasattr(self.model, "sample"):
            raise RuntimeError(_SAMPLER_UNAVAILABLE)
        embeddings = self.model.sample(n=int(n), seed=int(seed), condition=condition, **sample_kwargs)
        results = self._decode_embeddings(
            embeddings,
            source="sampled",
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            example_retrieval=example_retrieval,
        )
        return results[0] if n == 1 else results

    def _decode_sessions(
        self,
        sessions: list[list[dict[str, str]]],
        *,
        source: _Source,
        example_retrieval: bool,
        max_new_tokens: int | None,
        temperature: float | None,
        session_preprocess: str | None,
    ) -> list[PersonaManual]:
        bottleneck = self.model.encode(sessions, session_preprocess=session_preprocess or self.default_preprocess)
        return self._decode_embeddings(
            bottleneck,
            source=source,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            example_retrieval=example_retrieval,
        )

    def _decode_embeddings(
        self,
        bottleneck: torch.Tensor,
        *,
        source: _Source,
        max_new_tokens: int | None,
        temperature: float | None,
        example_retrieval: bool,
    ) -> list[PersonaManual]:
        config = getattr(self.model, "config", None)
        dual = bool(getattr(config, "dual_decode", False))
        slot = bool(getattr(config, "command_slot_decode", False))
        if dual and slot and hasattr(self.model, "generate_command_slot_blocks"):
            slot_tokens = 64 if max_new_tokens is None else int(max_new_tokens)
            temp = 0.0 if temperature is None else float(temperature)
            blocks = {
                head: self.model.generate_command_slot_blocks(
                    bottleneck,
                    head=head,
                    num_slots=num_slots,
                    max_new_tokens=slot_tokens,
                    temperature=temp,
                )
                for head, num_slots in (
                    ("general", int(getattr(config, "general_command_slots", 0) or 0)),
                    (
                        "specific",
                        int(getattr(config, "user_specific_command_slots", 0) or 0),
                    ),
                    ("style", int(getattr(config, "style_command_slots", 0) or 0)),
                )
            }
            manuals = [
                _manual_from_blocks(
                    general=blocks["general"][i],
                    specific=blocks["specific"][i],
                    style=blocks["style"][i],
                    embedding=bottleneck[i],
                    source=source,
                )
                for i in range(bottleneck.size(0))
            ]
            if example_retrieval:
                merged = [
                    {
                        "commands": m.commands,
                        "examples": m.examples,
                    }
                    for m in manuals
                ]
                if hasattr(self.model, "_inject_examples"):
                    merged = self.model._inject_examples(merged, bottleneck)
                    for manual, row in zip(manuals, merged, strict=True):
                        manual.examples = _examples(row)
                        manual.steering_prompt = render_steering(
                            general=manual.general,
                            specific=manual.specific,
                            style=manual.style,
                            examples=manual.examples,
                        )
            return manuals

        kwargs: dict[str, Any] = {"embeddings": bottleneck}
        if max_new_tokens is not None:
            kwargs["max_new_tokens"] = max_new_tokens
        if temperature is not None:
            kwargs["temperature"] = temperature
        if example_retrieval:
            kwargs["example_retrieval"] = True
        rows = self.model.generate_manual(**kwargs)
        return [_manual_from_merged(row, embedding=bottleneck[i], source=source) for i, row in enumerate(rows)]
