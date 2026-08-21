"""Unified CLI: ``python -m cue_training.baselines.main train <baseline> [...]``.

Baselines only *train* here; rollouts are generated through the real environment
(tau2 / simulatorarena) via the shared eval sidecar, which loads the trained
artifacts. The former offline ``rollout`` subcommand has been removed.

USP does not train locally in cue_training — ``train usp`` is a no-op that records
the published HuggingFace checkpoint.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from cue_training.baselines.common import dist as D
from cue_training.baselines.common.config import TrainConfig
from cue_training.baselines.common.registry import BASELINE_NAMES, get_baseline
from cue_training.utils.config import storage_root


def _add_common(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("baseline", choices=BASELINE_NAMES)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=storage_root() / "baselines",
        help="Where rollout JSONL and training artifacts are written.",
    )
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--device", default="cuda")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    train_p = sub.add_parser("train", help="Train a baseline (no-op where not needed).")
    _add_common(train_p)
    train_p.add_argument(
        "--data-path",
        type=Path,
        default=None,
        help="CUE preprocessed train.jsonl file or directory (defaults vary by baseline).",
    )
    train_p.add_argument(
        "--jsonl-names",
        nargs="+",
        default=["train.jsonl"],
        help="When --data-path is a directory, only read these filenames (default: train.jsonl), "
        "matching the encoder so the large train.unlabeled.jsonl is not included.",
    )
    train_p.add_argument("--model", default=None, help="Base model to train/adapt.")
    train_p.add_argument("--sim-model", default="gpt-4o-mini")
    train_p.add_argument("--api-key-env", default="OPENAI_API_KEY")
    train_p.add_argument("--api-base", default=None)
    train_p.add_argument("--epochs", type=int, default=1)
    train_p.add_argument(
        "--batch-size",
        type=int,
        default=1,
        help="Parallel LLM calls for data-prep (e.g. RealUserSim extraction).",
    )
    train_p.add_argument(
        "--sft-batch-size",
        type=int,
        default=1,
        help="Per-device SFT micro-batch size (unused for USP in cue_training).",
    )
    train_p.add_argument(
        "--sft-grad-accum",
        type=int,
        default=None,
        help="SFT gradient-accumulation steps (baseline-specific; unused for USP).",
    )
    train_p.add_argument(
        "--force-prep",
        action="store_true",
        help="Regenerate saved data-prep artifacts even if they already exist.",
    )

    return parser.parse_args(argv)


def _run_train(args: argparse.Namespace) -> None:
    # Init a process group when launched under torchrun so rank-0 data-prep can
    # synchronize before the DDP-aware HF/TRL trainers run on all GPUs.
    D.maybe_init_process_group()
    baseline = get_baseline(args.baseline)
    extra: dict[str, object] = {}
    if args.force_prep:
        extra["force_prep"] = True
    cfg = TrainConfig(
        output_dir=args.output_dir / args.baseline,
        data_path=args.data_path,
        model=args.model,
        sim_model=args.sim_model,
        api_key_env=args.api_key_env,
        api_base=args.api_base,
        device=args.device,
        limit=args.limit,
        epochs=args.epochs,
        batch_size=args.batch_size,
        sft_batch_size=args.sft_batch_size,
        sft_grad_accum=args.sft_grad_accum,
        jsonl_names=tuple(args.jsonl_names),
        extra=extra,
    )
    if D.is_main_process():
        cfg.output_dir.mkdir(parents=True, exist_ok=True)
    D.barrier()
    baseline.train(cfg)
    D.destroy()


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    _run_train(args)


if __name__ == "__main__":
    main()
