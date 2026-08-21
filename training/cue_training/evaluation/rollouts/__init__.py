"""Unified config-driven rollout orchestration for baseline and CUE modes."""

from __future__ import annotations

__all__ = ["main"]


def main(argv: list[str] | None = None) -> None:
    from cue_training.evaluation.rollouts.cli import main as cli_main

    cli_main(argv)
