"""Metrics command group."""

from __future__ import annotations

import argparse
import sys


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="cue-metrics")
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("single", help="Run metrics for one rollout")
    sub.add_parser("auto", help="Aggregate metrics across rollout outputs")
    args, rest = parser.parse_known_args(argv)
    if args.command == "single":
        from cue_training.evaluation.metrics.run import main as run_main
        sys.argv = [sys.argv[0], *rest]
        run_main()
    else:
        from cue_training.evaluation.metrics.aggregate import main as aggregate_main
        sys.argv = [sys.argv[0], *rest]
        aggregate_main()
