"""Process-global model singletons for evaluation harnesses.

Heavy models (CUE decoder, local base LMs) are loaded once and cached by
a strong key covering every behavior-affecting field, so paired/shuffled/mean and
different variants never accidentally share the wrong weights. All torch/transformers
imports are lazy so importing this module is cheap.
"""

from __future__ import annotations

from typing import Any

_CACHE: dict[tuple, Any] = {}


def _resolve_device(device: str) -> str:
    import torch

    if not (device.startswith("cuda") and torch.cuda.is_available()):
        return "cpu"
    # Guard an out-of-range ordinal: e.g. "cuda:1" when CUDA_VISIBLE_DEVICES exposes only one GPU
    # (remapped to cuda:0). Fall back to cuda:0 so loads / .to() / torch.load(map_location=...) don't
    # crash with "Attempting to deserialize object on CUDA device 1 but device_count() is 1".
    if ":" in device:
        try:
            if int(device.split(":", 1)[1]) >= torch.cuda.device_count():
                return "cuda:0"
        except ValueError:
            return "cuda:0"
    return device


def get_decoder(
    *, decoder_dir: str, device: str = "cuda", dtype: str = "bfloat16"
) -> Any:
    key = ("decoder", decoder_dir, device, dtype)
    if key in _CACHE:
        return _CACHE[key]

    from cue_training.infer.export import load_checkpoint

    dev = _resolve_device(device)
    model = load_checkpoint(decoder_dir, device=dev)
    # The conditioning modules compute in fp32 (GatedXAttnDense does h.float()); if a bf16
    # checkpoint load left their params bf16, the fp32 activations clash ("expected Float but
    # found BFloat16"). Force them back to fp32 (no-op if already fp32); the LM stays bf16.
    try:
        model.decoder.memory.float()
        model.decoder.gated_blocks.float()
        if model.decoder.slot_embeddings is not None:
            model.decoder.slot_embeddings.float()
            model.decoder.slot_projection.float()
    except AttributeError:
        pass
    model.eval()
    _CACHE[key] = model
    return model


def get_base_local(
    *, hf_path: str, device: str = "cuda", dtype: str = "bfloat16"
) -> dict[str, Any]:
    key = ("base_local", hf_path, device, dtype)
    if key in _CACHE:
        return _CACHE[key]

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    dev = _resolve_device(device)
    tokenizer = AutoTokenizer.from_pretrained(hf_path, trust_remote_code=True)
    tokenizer.padding_side = "left"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    lm = AutoModelForCausalLM.from_pretrained(
        hf_path, torch_dtype=getattr(torch, dtype), trust_remote_code=True, device_map=None
    ).to(dev)
    lm.eval()
    handle = {"lm": lm, "tokenizer": tokenizer, "device": dev}
    _CACHE[key] = handle
    return handle


def get_semantic(
    *, semantic_model: str = "intfloat/e5-base-v2", device: str = "cuda"
) -> dict[str, Any]:
    """Load the encoder's semantic model for embedding agent turns (decoder)."""

    key = ("semantic", semantic_model, device)
    if key in _CACHE:
        return _CACHE[key]

    from transformers import AutoModel, AutoTokenizer

    dev = _resolve_device(device)
    tokenizer = AutoTokenizer.from_pretrained(semantic_model)
    model = AutoModel.from_pretrained(semantic_model).to(dev).eval()
    handle = {"semantic_model": model, "semantic_tokenizer": tokenizer, "device": dev}
    _CACHE[key] = handle
    return handle


def get_baseline_simulator(
    *,
    name: str,
    output_dir: str,
    artifacts_dir: str | None = None,
    sim_model: str = "gpt-5.4-mini",
    device: str = "cuda",
    sim_api_base: str | None = None,
    sim_api_key_env: str = "OPENAI_API_KEY",
    sim_max_tokens: int = 512,
    temperature: float = 0.7,
    extra: dict[str, Any] | None = None,
) -> Any:
    """Build a baseline user simulator via the baselines registry (cached).

    ``extra`` is forwarded to ``RolloutConfig.extra`` so backends can opt into extra
    behavior (e.g. USP's vLLM serving via ``usp_vllm_base_url``/``usp_vllm_model``).
    """

    extra = extra or {}
    # The vLLM base_url distinguishes an HF-served sim from a vLLM-served one, so it must
    # be part of the cache key (otherwise the first-built variant would be reused).
    key = ("baseline", name, output_dir, artifacts_dir, sim_model, device,
           sim_api_base, sim_api_key_env,
           extra.get("usp_vllm_base_url"), extra.get("usp_vllm_model"),
           extra.get("userlm_vllm_base_url"), extra.get("userlm_vllm_model"))
    if key in _CACHE:
        return _CACHE[key]

    from pathlib import Path

    from cue_training.baselines.common.config import RolloutConfig
    from cue_training.baselines.common.registry import get_baseline

    cfg = RolloutConfig(
        output_dir=Path(output_dir),
        sim_model=sim_model,
        sim_api_base=sim_api_base,
        sim_api_key_env=sim_api_key_env,
        device=device,
        temperature=temperature,
        sim_max_tokens=sim_max_tokens,
        artifacts_dir=Path(artifacts_dir) if artifacts_dir else None,
        extra=dict(extra),
    )
    simulator = get_baseline(name).load(cfg)
    _CACHE[key] = simulator
    return simulator


def clear() -> None:
    _CACHE.clear()
