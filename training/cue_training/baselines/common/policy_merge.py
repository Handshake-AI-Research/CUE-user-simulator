"""Helpers for HF model dirs (policy-vLLM merge is unused; USP serves from HuggingFace)."""

from __future__ import annotations

from pathlib import Path

# Empty: USP loads ``wangkevin02/USP`` directly; no local adapter merge.
POLICY_VLLM_METHODS: frozenset[str] = frozenset()


def is_hf_model_dir(path: Path) -> bool:
    return path.is_dir() and (path / "config.json").is_file()


def ensure_merged_policy(method: str, artifact_root: Path, *, force: bool = False) -> Path:
    """Deprecated: cue_training does not merge local USP adapters."""

    raise ValueError(
        f"policy merge is not supported for {method!r}; "
        "USP serves the published HuggingFace checkpoint (wangkevin02/USP)."
    )
