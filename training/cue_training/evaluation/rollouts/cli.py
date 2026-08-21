"""Public ``cue-rollouts baseline|cue ...`` entrypoint."""

from __future__ import annotations

import argparse
import sys
from dataclasses import replace
from pathlib import Path

from cue_training.evaluation.rollouts.config import ConfigError, load_rollouts_config
from cue_training.evaluation.rollouts.orchestrator import Orchestrator
from cue_training.evaluation.rollouts.recipes import CUE_METHODS, BASELINE_METHODS, expand_methods


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="cue-rollouts",
        description=(
            "Unified rollout endpoint for baseline and CUE modes. "
            "Config selects benchmarks/simulators; methods select recipes."
        ),
    )
    p.add_argument(
        "rollout_type",
        choices=["baseline", "cue"],
        help="Rollout family.",
    )
    p.add_argument(
        "methods",
        nargs="+",
        help=(
            f"One or more methods, or 'all'. Baseline: {', '.join(BASELINE_METHODS)}. CUE: {', '.join(CUE_METHODS)}."
        ),
    )
    p.add_argument(
        "--config",
        default="configs/rollouts.json",
        help="Shared rollout config (default: configs/rollouts.json).",
    )
    p.add_argument(
        "--train",
        action="store_true",
        help="Baseline only: run paper-parity training before rollout.",
    )
    p.add_argument(
        "--force-train",
        action="store_true",
        help="Re-run paper training even when a complete manifest exists.",
    )
    p.add_argument(
        "--dry-run",
        action="store_true",
        help="Print resolved job DAG / GPU-port plan without side effects.",
    )
    p.add_argument(
        "--resume",
        action="store_true",
        help="Resume an existing run_id; requires matching config hash.",
    )
    p.add_argument(
        "--allow-hash-mismatch",
        action="store_true",
        help=(
            "With --resume, allow the config hash to differ from the run manifest "
            "(e.g. after bumping decode_batch_size). Keeps the stored hash so later "
            "resumes still match. Incomplete jobs re-run under the new knobs; COMPLETE "
            "jobs are still skipped."
        ),
    )
    p.add_argument("--limit", type=int, default=None, help="Max episodes/conversations per job.")
    p.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Replicate seed for sampled CUE conditions and baseline persona preparation.",
    )
    p.add_argument("--run-id", default=None, help="Override run id (default: UTC timestamp + config hash).")
    p.add_argument(
        "--force-prep",
        action="store_true",
        help="Rebuild tau2 cases (re-join CUE embeddings) and regenerate persona "
        "sidecars instead of reusing cached prep artifacts.",
    )
    p.add_argument(
        "--fresh-sidecar",
        action="store_true",
        help="Always start a new CUE sidecar (kill anything on its port) instead of reusing "
        "a running one -- avoids picking up a stale sidecar with old code.",
    )
    p.add_argument(
        "--arms",
        nargs="+",
        default=None,
        help=(
            "Override / subset of arms for this run (e.g. --arms sampled_pop). "
            "With --resume --run-id <existing>, replaces those arms in place and keeps "
            "the others. Use --force-prep to re-sample embeddings for the replaced arms."
        ),
    )
    return p


def _resolve_config_path(raw: str | Path) -> Path:
    path = Path(raw)
    if path.is_file():
        return path
    shipped = Path(__file__).resolve().parents[3] / path
    return shipped if shipped.is_file() else path


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    if args.train and args.rollout_type != "baseline":
        raise SystemExit("--train is only valid with rollout_type=baseline")
    if args.arms and args.resume and not args.run_id:
        raise SystemExit("--arms with --resume requires --run-id of the existing run")
    try:
        methods = expand_methods(args.rollout_type, list(args.methods))
        cfg_path = _resolve_config_path(args.config)
        cfg = load_rollouts_config(cfg_path)
        cfg = replace(cfg, cue=replace(cfg.cue, sampler_seed=int(args.seed)))
        orch = Orchestrator(
            cfg,
            rollout_type=args.rollout_type,
            methods=methods,
            train=bool(args.train or args.force_train),
            force_train=bool(args.force_train),
            dry_run=bool(args.dry_run),
            resume=bool(args.resume),
            limit=args.limit,
            run_id=args.run_id,
            force_prep=bool(args.force_prep),
            fresh_sidecar=bool(args.fresh_sidecar),
            arms=list(args.arms) if args.arms else None,
            seed=int(args.seed),
            allow_hash_mismatch=bool(args.allow_hash_mismatch),
        )
        result = orch.run()
        if not args.dry_run:
            print(f"run_id={result['run_id']} jobs={len(result.get('summaries') or [])}")
    except (ConfigError, ValueError) as exc:
        print(f"rollouts: {exc}", file=sys.stderr)
        raise SystemExit(2) from exc


if __name__ == "__main__":
    main()
