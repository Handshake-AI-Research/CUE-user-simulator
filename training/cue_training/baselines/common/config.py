"""Lightweight config objects shared by baseline ``train`` and ``rollout``."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class RolloutConfig:
    """Everything a baseline needs to produce rollout JSONL files."""

    output_dir: Path
    data_root: Path | None = None
    domain: str = "all"
    agent_model: str = "gpt-5.2"
    agent_api_key_env: str = "OPENAI_API_KEY"
    agent_api_base: str | None = None
    sim_model: str = "gpt-4o-mini"
    sim_api_key_env: str = "OPENAI_API_KEY"
    sim_api_base: str | None = None
    # Headroom for reasoning sim models (e.g. gpt-5.x), whose hidden reasoning
    # tokens count against the completion budget; 512 is too small and trips
    # "Could not finish the message because max_tokens ... was reached".
    sim_max_tokens: int = 2048
    # Optional override for persona-induction models; defaults to ``sim_model`` when unset.
    persona_model: str | None = None
    device: str = "cuda"
    limit: int | None = None
    max_turns: int | None = None
    temperature: float = 0.7
    batch_size: int = 1  # >1 runs episodes concurrently (API-based simulators)
    artifacts_dir: Path | None = None
    extra: dict[str, Any] = field(default_factory=dict)

    def baseline_dir(self, baseline_name: str) -> Path:
        """Directory where rollout files for a baseline are written."""

        return self.output_dir / baseline_name

    def artifacts_path(self, baseline_name: str) -> Path:
        """Directory where a baseline's training artifacts are read from."""

        return self.artifacts_dir or self.baseline_dir(baseline_name)


@dataclass
class TrainConfig:
    """Everything a baseline needs to (optionally) train."""

    output_dir: Path
    data_path: Path | None = None
    model: str | None = None
    sim_model: str = "gpt-4o-mini"
    api_key_env: str = "OPENAI_API_KEY"
    api_base: str | None = None
    device: str = "cuda"
    limit: int | None = None
    epochs: int = 1
    batch_size: int = 1  # >1 parallelizes LLM data-prep calls (e.g. extraction)
    # GPU SFT micro-batch + grad-accum (usp); distinct from batch_size (which
    # is API concurrency). sft_grad_accum None keeps the baseline-specific default.
    sft_batch_size: int = 1
    sft_grad_accum: int | None = None
    # Labeled training filenames to read when ``data_path`` is a directory; mirrors
    # the encoder's ``--jsonl_names`` so the unlabeled corpus isn't swept in.
    jsonl_names: tuple[str, ...] = ("train.jsonl",)
    extra: dict[str, Any] = field(default_factory=dict)

    def artifacts_path(self) -> Path:
        """Training artifacts are written directly under this baseline-scoped dir."""

        return self.output_dir
