"""Prep: build tau2 eval cases from tau_usi normalized data + CUE embeddings.

Reuses the CUE encoder export (per-episode ``trajectory_embedding``) rather than the
MirrorBench payloads path; the run script exports embeddings first, then calls this.
Prints a mapping report (mapped/skipped counts) as the first smoke artifact.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from cue_training.evaluation.common.io import write_jsonl
from cue_training.evaluation.common.sampler_arms import add_sampler_argparse, sampler_kwargs_from_args
from cue_training.evaluation.tau2_bench.data import CUE_ARMS, build_cases, load_episodes, resolve_tau2_task_ids
from cue_training.runlog.log import log, warn


def run_prep(args: argparse.Namespace) -> None:
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    episodes = load_episodes(args.normalized)
    # Content-match each tau-usi episode to the CORRECT tau2 task id (the numeric suffix is NOT
    # the tau2 task id). Without --tau2_path we can't match, so fail loudly rather than silently
    # running mismatched tasks.
    if args.tau2_path:
        map_counts = resolve_tau2_task_ids(episodes, args.tau2_path, min_score=args.match_min_score)
        log("tau2:prep", f"tau2 task-id content match: {json.dumps(map_counts)}")
    else:
        warn(
            "tau2:prep",
            "--tau2_path not set; falling back to the (unreliable) numeric-suffix task id. "
            "Pass --tau2_path to content-match the correct tau2 task.",
        )
    cases, counts = build_cases(
        episodes, args.embeddings, arms=tuple(args.arms), seed=args.seed,
        require_embeddings=args.require_embeddings,
        **{k: v for k, v in sampler_kwargs_from_args(args).items()},
    )

    for arm, arm_cases in cases.items():
        write_jsonl(str(out_dir / f"cases.{arm}.jsonl"), arm_cases)
    (out_dir / "prep_counts.json").write_text(json.dumps(counts, indent=2), encoding="utf-8")
    log("tau2:prep", json.dumps(counts))
    for arm, arm_cases in cases.items():
        log("tau2:prep", f"arm {arm}: {len(arm_cases)} cases -> {out_dir / f'cases.{arm}.jsonl'}")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Build tau2 eval cases (customer-service/tau_usi).")
    p.add_argument("--normalized", required=True, help="tau_usi normalized.jsonl")
    p.add_argument("--embeddings", required=True, help="CUE encoder export JSON over tau_usi eval")
    p.add_argument("--out_dir", required=True)
    p.add_argument("--arms", nargs="+", default=list(CUE_ARMS))
    p.add_argument("--tau2_path", default=None,
                   help="tau2-bench repo path; used to content-match tau-usi episodes to the correct "
                        "tau2 task id (the instance-id numeric suffix is NOT the tau2 task id).")
    p.add_argument("--match_min_score", type=float, default=0.5,
                   help="Min token-Jaccard for a confident tau-usi->tau2 task match (else the episode "
                        "is skipped).")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument(
        "--require_embeddings",
        type=lambda v: str(v).lower() not in ("0", "false", "no"),
        default=True,
        help="Require a joined CUE embedding per case (CUE/retrieval). Set 0 for persona/"
             "as_is baselines whose arms select a profile, not an embedding.",
    )
    add_sampler_argparse(p)
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    run_prep(parse_args(argv))
