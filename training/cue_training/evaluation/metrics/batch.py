"""Run many metrics evals in ONE process so the OSS model is shared across all rollouts.

Each ``metrics`` invocation loads the OSS Qwen model (module-level lru_cache) once, but a
separate process per (baseline, candidate) reloads it every time. This driver loops over a
jobs file in a single process, so the model + fitted baseline state are reused across every
rollout eval. Baseline state is still deduped per baseline via the cache manifest.

Jobs file: one job per line, ``baseline_rollout[<TAB>candidate_rollout]`` (candidate omitted
= score the baseline against its own human reference / fit its state).
"""

from __future__ import annotations

import argparse
from pathlib import Path

from cue_training.evaluation.metrics.run import _parse_metrics, run_one
from cue_training.runlog.log import log, warn


def _read_jobs(path: str | Path) -> list[tuple[Path, Path | None]]:
    jobs: list[tuple[Path, Path | None]] = []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        parts = line.split("\t")
        baseline = Path(parts[0])
        candidate = Path(parts[1]) if len(parts) > 1 and parts[1].strip() else None
        jobs.append((baseline, candidate))
    return jobs


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description="Batch-run metrics evals in one process (shared OSS model).")
    p.add_argument("--jobs", required=True, help="Jobs file: 'baseline[<TAB>candidate]' per line.")
    p.add_argument("--metrics", default="default", help="'default', 'all', or supported metric names/aliases.")
    p.add_argument("--oss-model", default="Qwen/Qwen3-8B")
    p.add_argument("--tau-usi-source", default=None)
    p.add_argument("--refit", action="store_true", help="Rebuild baseline state even if the cache matches.")
    p.add_argument("--no-fit", action="store_true", help="Candidates reuse existing baseline state only (fail if missing); baseline-only jobs still fit.")
    p.add_argument(
        "--merge",
        action="store_true",
        help="Patch only --metrics into each job's existing summary.json (keep other metrics).",
    )
    args = p.parse_args(argv)

    metrics = _parse_metrics(args.metrics)
    jobs = _read_jobs(args.jobs)
    log("metrics:batch", f"{len(jobs)} job(s) in one process (metrics={args.metrics})")

    n_ok = n_err = 0
    for i, (baseline, candidate) in enumerate(jobs, 1):
        label = baseline.parent.name + (f" -> {candidate.parent.name}" if candidate else " (baseline-only)")
        log("metrics:batch", f"[{i}/{len(jobs)}] {label}")
        try:
            run_one(
                baseline, candidate, metrics=metrics, oss_model=args.oss_model,
                tau_usi_source=args.tau_usi_source, refit=args.refit,
                no_fit=bool(args.no_fit and candidate is not None),
                merge=args.merge,
            )
            n_ok += 1
        except Exception as exc:  # noqa: BLE001 - one bad rollout shouldn't kill the batch
            warn("metrics:batch", f"job failed ({label}): {exc}")
            n_err += 1
    log("metrics:batch", f"done: {n_ok} ok, {n_err} failed")


if __name__ == "__main__":
    main()
