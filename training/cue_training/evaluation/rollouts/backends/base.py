"""Shared backend protocol and helpers."""

from __future__ import annotations

import json
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from cue_training.evaluation.rollouts.artifacts import JobSpec, RunPaths, mark_job_complete, summarize_results, write_job_manifest
from cue_training.evaluation.rollouts.config import RolloutsConfig


@dataclass
class BackendContext:
    cfg: RolloutsConfig
    paths: RunPaths
    job: JobSpec
    job_dir: Path
    dry_run: bool = False
    resume: bool = False
    limit: int | None = None
    vllm_urls: dict[str, str] | None = None
    sidecar_url: str | None = None
    work_dir: Path | None = None
    force_prep: bool = False
    # When set, rerun only these arms and merge into existing job artifacts in place.
    replace_arms: list[str] | None = None

    @property
    def scratch_dir(self) -> Path:
        """Bulky-intermediates dir; falls back to job_dir when no scratch root is set."""

        return self.work_dir or self.job_dir


class BenchmarkBackend(ABC):
    name: str

    @abstractmethod
    def prepare(self, ctx: BackendContext) -> dict[str, Any]:
        raise NotImplementedError

    @abstractmethod
    def run(self, ctx: BackendContext, prep: dict[str, Any]) -> dict[str, Any]:
        raise NotImplementedError

    def finalize(self, ctx: BackendContext, run_info: dict[str, Any]) -> dict[str, Any]:
        summary = summarize_results(ctx.job_dir / "results.jsonl")
        summary.update(run_info)
        (ctx.job_dir / "summary.json").write_text(
            json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        write_job_manifest(
            ctx.job_dir,
            {
                "job": ctx.job.__dict__,
                "prep": run_info.get("prep") or {},
                "shards": run_info.get("shards") or [],
                "summary": summary,
                "status": "complete",
            },
        )
        mark_job_complete(ctx.job_dir)
        return summary


def get_backend(benchmark: str) -> BenchmarkBackend:
    if benchmark == "tau2":
        from cue_training.evaluation.rollouts.backends.tau2 import Tau2Backend

        return Tau2Backend()
    if benchmark == "simulatorarena":
        from cue_training.evaluation.rollouts.backends.simarena import SimArenaBackend

        return SimArenaBackend()
    if benchmark == "prism":
        from cue_training.evaluation.rollouts.backends.prism import PrismBackend

        return PrismBackend()
    raise ValueError(f"unknown benchmark {benchmark!r}")
