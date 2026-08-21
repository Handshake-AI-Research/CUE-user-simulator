"""Export a trained checkpoint into the sibling `cue-hf` inference package layout.

`cue-hf` reads a plain HF-style directory: ``config.json`` (``model_type: cue``) plus one
weight file holding the whole model. The encoder/decoder *tokenizers* are not copied —
cue-hf pulls them from the base repos named in the config.

The example pool ships separately as a Hugging Face **dataset**, because it is far larger
than the model and most callers do not need retrieval.
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Any, Iterator

from cue_training.infer.export import load_checkpoint
from cue_training.utils.log import log

TAG = "export_cue_hf"
CONFIG_NAME = "config.json"
WEIGHTS_NAME = "model.safetensors"
SAMPLER_NAME = "sampler.pt"


def cue_hf_config(
    model: Any,
    *,
    session_preprocess: str | None = None,
    example_pool_id: str | None = None,
    sampler_id: str | None = None,
) -> dict[str, Any]:
    """Architecture keys for CueConfig. Decode/retrieval knobs use cue-hf defaults."""

    config = model.config_dict()
    payload: dict[str, Any] = {
        "model_type": "cue",
        "architectures": ["CueModel"],
        "encoder": config["encoder"],
        "decoder": dict(config["decoder"]),
        "dual_decode": bool(config["dual_decode"]),
        "command_slot_decode": bool(config["command_slot_decode"]),
        "general_command_slots": int(config["general_command_slots"]),
        "user_specific_command_slots": int(config["user_specific_command_slots"]),
        "style_command_slots": int(config["style_command_slots"]),
        "slot_embedding_dim": int(config["slot_embedding_dim"]),
    }
    # config_dict() records the built decoder, which omits the dtype it was built with.
    payload["decoder"].setdefault("dtype", "bfloat16")
    if session_preprocess:
        payload["session_preprocess"] = session_preprocess
    if example_pool_id:
        payload["example_pool_id"] = example_pool_id
    if sampler_id:
        payload["sampler_id"] = sampler_id
    return payload


def export_cue_hf(
    checkpoint: str | Path,
    output: str | Path,
    *,
    device: str = "cpu",
    sampler_checkpoint: str | Path | None = None,
    example_pool_id: str | None = None,
    sampler_id: str | None = None,
    session_preprocess: str | None = None,
    push_to: str | None = None,
    private: bool = True,
) -> Path:
    """Write a cue-hf-loadable model directory for a joint or refinement checkpoint."""

    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    model = load_checkpoint(checkpoint, device=device)
    payload = cue_hf_config(
        model,
        session_preprocess=session_preprocess,
        example_pool_id=example_pool_id,
        sampler_id=sampler_id,
    )
    (output / CONFIG_NAME).write_text(
        json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8"
    )
    _save_safetensors(model, output / WEIGHTS_NAME)
    if sampler_checkpoint:
        source = Path(sampler_checkpoint)
        if source.is_dir():
            source = source / SAMPLER_NAME
        shutil.copyfile(source, output / SAMPLER_NAME)
        log(TAG, f"copied sampler {source} -> {output / SAMPLER_NAME}")
        import torch

        if "optimizer" in torch.load(source, map_location="cpu", mmap=True, weights_only=False):
            log(
                TAG,
                f"{SAMPLER_NAME} still holds training state (optimizer/EMA); every caller "
                "downloads it. Shrink it ~4x with `cue-hf slim-sampler`.",
            )
    log(TAG, f"exported cue-hf model -> {output}")
    if push_to:
        from huggingface_hub import HfApi

        api = HfApi()
        api.create_repo(push_to, private=private, exist_ok=True)
        api.upload_folder(repo_id=push_to, folder_path=str(output))
        log(TAG, f"pushed cue-hf model -> {push_to}")
    return output


def _save_safetensors(model: Any, path: Path) -> None:
    """Write the full state dict, dropping the tied output embedding.

    safetensors refuses aliased tensors. The *input* embedding is the one to keep: cue-hf
    re-ties on load, which points lm_head at whatever the input embedding holds, so
    shipping lm_head instead would silently restore random embeddings.
    """

    from safetensors.torch import save_file

    state = model.state_dict()
    lm = model.decoder.lm
    output_embeddings = getattr(lm, "get_output_embeddings", lambda: None)()
    tied_ptr = None
    input_key = ""
    if output_embeddings is not None and hasattr(output_embeddings, "weight"):
        input_weight = lm.get_input_embeddings().weight
        if output_embeddings.weight.data_ptr() == input_weight.data_ptr():
            tied_ptr = input_weight.data_ptr()
            # state_dict() detaches, so find the canonical name among the parameters.
            name = next(n for n, p in lm.named_parameters() if p is input_weight)
            input_key = f"decoder.lm.{name}"
    kept = {}
    for key, value in state.items():
        if tied_ptr is not None and value.data_ptr() == tied_ptr and key != input_key:
            log(TAG, f"dropping tied weight {key} (aliases {input_key})")
            continue
        kept[key] = value.contiguous()
    save_file(kept, str(path), metadata={"format": "pt"})


def _pool_rows(pool: Any) -> Iterator[dict[str, Any]]:
    for session_id, vector, examples in zip(
        pool.session_ids, pool.cue_vectors, pool.examples, strict=True
    ):
        yield {
            "session_id": session_id,
            "cue_embedding": vector.astype("float32").tolist(),
            "examples": [{"text": e.text, "kind": e.kind} for e in examples],
        }


def export_example_pool_dataset(
    checkpoint: str | Path,
    *,
    output: str | Path | None = None,
    push_to: str | None = None,
    private: bool = True,
) -> Any:
    """Turn a built example pool into a HF dataset for cue-hf's retrieval path."""

    from datasets import Dataset

    from cue_training.data.example_pool import load_pool_for_checkpoint

    pool = load_pool_for_checkpoint(checkpoint)
    if pool is None:
        raise FileNotFoundError(
            f"{checkpoint} has no example_pool/; build one with `cue build-example-pool`"
        )
    dataset = Dataset.from_generator(lambda: _pool_rows(pool))
    dataset.info.description = (
        "CUE example pool for cue-hf retrieval.\n"
        f"checkpoint_fingerprint: {pool.checkpoint_fingerprint}\n"
        f"dim: {pool.dim}\n"
    )
    if output:
        dataset.save_to_disk(str(output))
        log(TAG, f"wrote example-pool dataset -> {output}")
    if push_to:
        dataset.push_to_hub(push_to, private=private)
        log(TAG, f"pushed example-pool dataset -> {push_to}")
    return dataset
