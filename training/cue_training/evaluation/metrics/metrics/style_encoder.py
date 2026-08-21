from __future__ import annotations

import os
from functools import lru_cache
from typing import Any

import numpy as np

DEFAULT_STYLE_MODEL = "AnnaWegmann/Style-Embedding"
DEFAULT_STYLEDISTANCE_MODEL = "StyleDistance/styledistance"
DEFAULT_OSS_MODEL = "Qwen/Qwen3-8B"

# MirrorBench classifier/oss_hidden_state_probe (mirrorbench/metrics/classifier/_shared.py).
_OSS_QUERY = "Is the following text written by a real human user rather than a user simulator?"
# Representation query for the style/self-similarity + MAUVE metrics: instead of the
# discriminative "is this human?" prompt, seed the Answer: hidden state with a request to
# characterize the user's style, so the pooled vector encodes behavioral/stylistic markers.
_OSS_STYLE_QUERY = "What are the behavioral and stylistic markers of the user in the following trajectory?"
_OSS_SEED_TOKEN = "Answer:"
_OSS_INPUT_TEMPLATE = "Query: {query}\nText: {text}\n{seed_token}"
_OSS_LAYER = -1
# Cache/schema tags so features built with a different query/pooling are not mixed.
OSS_FEATURE_KIND = "query_text_answer_last_token"
OSS_STYLE_FEATURE_KIND = "query_style_markers_answer_last_token"


def pick_device(explicit: str | None = None) -> str:
    """CUDA, else Apple MPS, else CPU. Shared by the HF and LUAR encoders."""

    import torch

    if explicit:
        return explicit
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def _format_oss_probe_input(text: str, query: str = _OSS_QUERY) -> str:
    formatted = _OSS_INPUT_TEMPLATE.format(
        query=query, text=text or "", seed_token=_OSS_SEED_TOKEN
    )
    if not formatted.rstrip().endswith(_OSS_SEED_TOKEN):
        formatted = f"{formatted.rstrip()}\n{_OSS_SEED_TOKEN}"
    return formatted


class HFEncoder:
    """Mean-pooled hidden-state encoder over a HuggingFace model.

    Used by the style-embedding metrics (``AnnaWegmann/Style-Embedding``): mean-pool the
    last hidden state with the attention mask. Torch/transformers are imported lazily and
    the model runs on CUDA when available.
    """

    def __init__(
        self,
        model_name: str,
        *,
        device: str | None = None,
        batch_size: int = 16,
        dtype: str | None = None,
        trust_remote_code: bool = False,
    ) -> None:
        import torch
        from transformers import AutoModel, AutoTokenizer

        self.model_name = model_name
        self.batch_size = batch_size
        self.device = pick_device(device)
        torch_dtype = getattr(torch, dtype) if dtype else None
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=trust_remote_code)
        self.model = (
            AutoModel.from_pretrained(model_name, torch_dtype=torch_dtype, trust_remote_code=trust_remote_code)
            .to(self.device)
            .eval()
        )

    def encode(self, texts: list[str]) -> np.ndarray:
        import torch

        vectors: list[np.ndarray] = []
        for start in range(0, len(texts), self.batch_size):
            chunk = [t or "" for t in texts[start : start + self.batch_size]]
            enc = self.tokenizer(chunk, padding=True, truncation=True, max_length=512, return_tensors="pt").to(self.device)
            with torch.no_grad():
                out = self.model(**enc).last_hidden_state
            mask = enc["attention_mask"].unsqueeze(-1).to(out.dtype)
            pooled = (out * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1e-9)
            vectors.append(pooled.cpu().float().numpy())
        if not vectors:
            return np.zeros((0, 1), dtype=float)
        return np.concatenate(vectors, axis=0)


@lru_cache(maxsize=2)
def _load_oss_model(
    model_name: str = DEFAULT_OSS_MODEL,
    dtype: str = "bfloat16",
    device: str | None = None,
    trust_remote_code: bool = True,
) -> tuple[Any, Any, str]:
    """Load (and cache) the OSS CausalLM + tokenizer once per (model, dtype, device).

    Every OSS encoder wrapper reuses this, so all OSS metrics (probe / style-representation /
    profile generation) share a SINGLE model copy in memory. Returns (tokenizer, model, device)."""

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    dev = device or ("cuda" if torch.cuda.is_available() else "cpu")
    torch_dtype = getattr(torch, dtype) if dtype else None
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=trust_remote_code)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    # Right pad + left truncate: last real token stays ``Answer:`` under both padding and
    # length limits (matches MirrorBench pooling; left truncate is the safe long-traj fix).
    tokenizer.padding_side = "right"
    tokenizer.truncation_side = "left"
    model = (
        AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch_dtype, trust_remote_code=trust_remote_code)
        .to(dev)
        .eval()
    )
    return tokenizer, model, dev


class OSSProbeEncoder:
    """MirrorBench OSS hidden-state probe encoder.

    Each text (episode = joined USER turns) is wrapped as
    ``Query: …\\nText: {trajectory}\\nAnswer:``; the feature is the CausalLM hidden state
    at the ``Answer:`` (last non-pad) token, then L2-normalized. Left truncation keeps
    ``Answer:`` when the trajectory exceeds ``max_length``. The heavy model/tokenizer are
    shared across wrappers via :func:`_load_oss_model` (one copy for all OSS metrics).
    """

    def __init__(
        self,
        model_name: str = DEFAULT_OSS_MODEL,
        *,
        device: str | None = None,
        batch_size: int = 4,
        dtype: str = "bfloat16",
        max_length: int = 512,
        layer: int = _OSS_LAYER,
        trust_remote_code: bool = True,
        query: str = _OSS_QUERY,
        feature_kind: str = OSS_FEATURE_KIND,
    ) -> None:
        self.model_name = model_name
        self.batch_size = batch_size
        self.max_length = max_length
        self.layer = layer
        self.query = query
        self.feature_kind = feature_kind
        # The heavy model+tokenizer are loaded once per (model, dtype, device) and SHARED across
        # every OSS encoder wrapper (discriminative probe, style-representation, profile generator)
        # -- they differ only by query/feature_kind, so one Qwen-8B copy serves all OSS metrics.
        self.tokenizer, self.model, self.device = _load_oss_model(model_name, dtype, device, trust_remote_code)

    def generate_text(
        self, prompts: list[str], *, max_new_tokens: int = 300, temperature: float = 0.0,
        max_input_tokens: int = 3072,
    ) -> list[str]:
        """Chat-generate a completion per prompt with the same CausalLM used for the probe.

        Used for the LLM style-profile classifier so the profile is written by the OSS model
        rather than an external API. Batched with left padding; Qwen ``thinking`` is disabled so
        the output is the profile text only."""

        import torch

        outs: list[str] = []
        prev_pad = self.tokenizer.padding_side
        self.tokenizer.padding_side = "left"  # decoder-only batched generation needs left padding
        try:
            for start in range(0, len(prompts), self.batch_size):
                chunk = prompts[start : start + self.batch_size]
                rendered: list[str] = []
                for p in chunk:
                    messages = [{"role": "user", "content": p}]
                    try:
                        text = self.tokenizer.apply_chat_template(
                            messages, tokenize=False, add_generation_prompt=True, enable_thinking=False
                        )
                    except TypeError:  # tokenizer without enable_thinking kwarg
                        text = self.tokenizer.apply_chat_template(
                            messages, tokenize=False, add_generation_prompt=True
                        )
                    rendered.append(text)
                enc = self.tokenizer(
                    rendered, padding=True, truncation=True, max_length=max_input_tokens, return_tensors="pt"
                ).to(self.device)
                with torch.no_grad():
                    gen = self.model.generate(
                        **enc, max_new_tokens=max_new_tokens,
                        do_sample=temperature > 0, temperature=temperature if temperature > 0 else None,
                        pad_token_id=self.tokenizer.pad_token_id,
                    )
                new = gen[:, enc["input_ids"].shape[1]:]
                outs.extend(t.strip() for t in self.tokenizer.batch_decode(new, skip_special_tokens=True))
        finally:
            self.tokenizer.padding_side = prev_pad
        return outs

    def encode(self, texts: list[str]) -> np.ndarray:
        import torch

        vectors: list[np.ndarray] = []
        for start in range(0, len(texts), self.batch_size):
            raw = texts[start : start + self.batch_size]
            batch = [_format_oss_probe_input(t, self.query) for t in raw]
            enc = self.tokenizer(
                batch,
                padding=True,
                truncation=True,
                max_length=self.max_length,
                return_tensors="pt",
            ).to(self.device)
            with torch.no_grad():
                out = self.model(**enc, output_hidden_states=True)
            hidden = out.hidden_states[self.layer]  # [B, T, H]
            last = enc["attention_mask"].sum(dim=1).clamp(min=1) - 1
            pooled = hidden[torch.arange(hidden.size(0), device=hidden.device), last]
            vecs = torch.nn.functional.normalize(pooled.float(), dim=-1)
            vectors.append(vecs.cpu().numpy())
        if not vectors:
            hidden_size = int(getattr(self.model.config, "hidden_size", 1))
            return np.zeros((0, hidden_size), dtype=np.float32)
        return np.concatenate(vectors, axis=0)


class OSSRemoteEncoder:
    """Same Answer:-token features as ``OSSProbeEncoder`` but served by a vLLM pooling/embed
    endpoint (``--convert embed`` -> ``/v1/embeddings``: LAST pooling + L2-normalize, which is
    exactly this feature). Lets many eval processes share ONE GPU-hosted model in parallel.

    Enabled by MIRROR_OSS_EMBED_URL; the trajectory is wrapped with the same query template and
    POSTed, so the returned vector matches the in-process encoder's (fit + score must both use
    this backend -- the state_config carries an oss_backend tag to force a refit on switch)."""

    def __init__(
        self, base_url: str, *, served_model: str, api_key_env: str = "HOSTED_VLLM_API_KEY",
        query: str = _OSS_QUERY, feature_kind: str = OSS_FEATURE_KIND, max_length: int = 512,
        batch_size: int = 16, timeout: float = 600.0,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.served_model = served_model
        self.api_key_env = api_key_env
        self.query = query
        self.feature_kind = feature_kind
        self.max_length = max_length
        self.batch_size = batch_size
        self.timeout = timeout

    def encode(self, texts: list[str]) -> np.ndarray:
        import json
        import os
        import urllib.request

        if not texts:
            return np.zeros((0, 1), dtype=np.float32)
        api_key = os.environ.get(self.api_key_env, "") or "EMPTY"
        vectors: list[list[float]] = []
        for start in range(0, len(texts), self.batch_size):
            chunk = [_format_oss_probe_input(t, self.query) for t in texts[start : start + self.batch_size]]
            body = {"model": self.served_model, "input": chunk, "truncate_prompt_tokens": self.max_length}
            req = urllib.request.Request(
                f"{self.base_url}/embeddings",
                data=json.dumps(body).encode("utf-8"),
                headers={"Content-Type": "application/json", "Authorization": f"Bearer {api_key}"},
                method="POST",
            )
            with urllib.request.urlopen(req, timeout=self.timeout) as resp:  # noqa: S310
                out = json.loads(resp.read().decode("utf-8"))
            rows = sorted(out["data"], key=lambda d: d.get("index", 0))
            vectors.extend(r["embedding"] for r in rows)
        return np.asarray(vectors, dtype=np.float32)


def _oss_embed_url() -> str | None:
    return os.environ.get("MIRROR_OSS_EMBED_URL") or None


@lru_cache(maxsize=4)
def get_encoder(
    model_name: str = DEFAULT_STYLE_MODEL,
    *,
    dtype: str | None = None,
    trust_remote_code: bool = False,
    batch_size: int = 16,
) -> Any:
    """Cached mean-pool encoder (style embedding); raises if torch/transformers unavailable."""

    return HFEncoder(model_name, dtype=dtype, trust_remote_code=trust_remote_code, batch_size=batch_size)


@lru_cache(maxsize=2)
def get_styledistance_encoder(
    model_name: str = DEFAULT_STYLEDISTANCE_MODEL,
    *,
    batch_size: int = 16,
) -> Any:
    """StyleDistance content-independent style embeds (one vector per concatenated text)."""

    return HFEncoder(model_name, batch_size=batch_size, trust_remote_code=True)


def _oss_encoder(model_name: str, dtype: str, batch_size: int, max_length: int, query: str, feature_kind: str) -> Any:
    """Remote vLLM-embed encoder when MIRROR_OSS_EMBED_URL is set, else the in-process HF probe."""

    url = _oss_embed_url()
    if url:
        return OSSRemoteEncoder(
            url, served_model=os.environ.get("MIRROR_OSS_EMBED_MODEL", model_name),
            api_key_env=os.environ.get("MIRROR_OSS_EMBED_API_KEY_ENV", "HOSTED_VLLM_API_KEY"),
            query=query, feature_kind=feature_kind, max_length=max_length, batch_size=batch_size,
        )
    return OSSProbeEncoder(
        model_name, dtype=dtype, trust_remote_code=True, batch_size=batch_size, max_length=max_length,
        query=query, feature_kind=feature_kind,
    )


@lru_cache(maxsize=2)
def get_oss_probe_encoder(
    model_name: str = DEFAULT_OSS_MODEL,
    *,
    dtype: str = "bfloat16",
    batch_size: int = 4,
    max_length: int = 512,
) -> Any:
    """OSS probe encoder (discriminative Query/Text/Answer: last-token features). vLLM-embed when
    MIRROR_OSS_EMBED_URL is set (shareable across processes), else the shared in-process HF model."""

    return _oss_encoder(model_name, "bfloat16", batch_size, max_length, _OSS_QUERY, OSS_FEATURE_KIND)


@lru_cache(maxsize=2)
def get_oss_style_encoder(
    model_name: str = DEFAULT_OSS_MODEL,
    *,
    dtype: str = "bfloat16",
    batch_size: int = 4,
    max_length: int = 512,
) -> Any:
    """OSS *style-representation* encoder: same Answer:-token pooling as the probe but seeded with
    the behavioral/stylistic-markers query, for self-similarity + MAUVE cluster."""

    return _oss_encoder(model_name, "bfloat16", batch_size, max_length, _OSS_STYLE_QUERY, OSS_STYLE_FEATURE_KIND)


@lru_cache(maxsize=1)
def get_oss_hf_generator(
    model_name: str = DEFAULT_OSS_MODEL, *, dtype: str = "bfloat16", batch_size: int = 4, max_length: int = 512,
) -> OSSProbeEncoder:
    """In-process HF OSS model for style-profile generation (fallback when no gen vLLM URL).

    Kept separate from the embed factories so generation never depends on the embed backend
    (which, under MIRROR_OSS_EMBED_URL, returns a remote encoder with no generate_text)."""

    return OSSProbeEncoder(model_name, dtype=dtype, trust_remote_code=True, batch_size=batch_size, max_length=max_length)
