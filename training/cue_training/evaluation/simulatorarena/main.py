"""Unified ``simarena-eval`` CLI: ``simarena-eval {prep, run} ...``.

``run`` builds the doc-creation cases (join to CUE writing embeddings) and runs the
conversation loop per variant/arm. ``prep`` only builds + writes the cases for inspection.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def _prep(argv: list[str]) -> None:
    from cue_training.evaluation.common.io import write_jsonl
    from cue_training.evaluation.common.sampler_arms import add_sampler_argparse, sampler_kwargs_from_args
    from cue_training.evaluation.simulatorarena.data import CUE_ARMS, build_cases, load_annotations

    p = argparse.ArgumentParser(description="Build SimulatorArena cases (writing/math).")
    p.add_argument("--annotations", required=True)
    p.add_argument("--embeddings", required=True)
    p.add_argument("--out_dir", required=True)
    p.add_argument("--domain", choices=["writing", "math"], default="writing")
    p.add_argument("--background", default=None)
    p.add_argument("--profiles_dir", default=None)
    p.add_argument("--arms", nargs="+", default=list(CUE_ARMS))
    p.add_argument("--seed", type=int, default=0)
    add_sampler_argparse(p)
    args = p.parse_args(argv)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    cases, counts = build_cases(
        load_annotations(args.annotations), args.embeddings, arms=tuple(args.arms),
        background_path=args.background, profiles_dir=args.profiles_dir, seed=args.seed,
        domain=args.domain,
        **sampler_kwargs_from_args(args),
    )
    for arm, arm_cases in cases.items():
        write_jsonl(str(out_dir / f"cases.{arm}.jsonl"), arm_cases)
    (out_dir / "prep_counts.json").write_text(json.dumps(counts, indent=2), encoding="utf-8")
    from cue_training.runlog.log import log

    log("simarena:prep", json.dumps(counts))


def _normalized(argv: list[str]) -> None:
    """Write MirrorBench-style normalized.jsonl from SimulatorArena annotations (for the CUE
    encoder embedding export). Needed for the math domain (writing already ships one)."""

    from cue_training.evaluation.common.io import write_jsonl
    from cue_training.evaluation.simulatorarena.data import load_annotations, to_normalized_rows

    p = argparse.ArgumentParser(description="Build MirrorBench normalized.jsonl from simarena annotations.")
    p.add_argument("--annotations", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--domain", choices=["writing", "math"], default="math")
    p.add_argument("--background", default=None)
    args = p.parse_args(argv)

    rows = to_normalized_rows(load_annotations(args.annotations), domain=args.domain,
                              background_path=args.background)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    write_jsonl(args.out, rows)
    from cue_training.runlog.log import log

    log("simarena:normalized", f"wrote {len(rows)} rows ({args.domain}) -> {args.out}")


def _usage() -> str:
    return "usage: cue-simarena-eval {prep,run,normalized} ...\n"


def main(argv: list[str] | None = None) -> None:
    argv = list(sys.argv[1:] if argv is None else argv)
    if not argv or argv[0] in ("-h", "--help"):
        sys.stdout.write(_usage())
        return
    command, rest = argv[0], argv[1:]
    if command == "prep":
        _prep(rest)
    elif command == "normalized":
        _normalized(rest)
    elif command == "run":
        from cue_training.evaluation.simulatorarena.runner import main as run_main

        run_main(rest)
    else:
        sys.stderr.write(f"simarena-eval: unknown command {command!r}\n{_usage()}")
        raise SystemExit(2)


if __name__ == "__main__":
    main()
