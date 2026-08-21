"""Data preparation command group."""

from __future__ import annotations

import argparse


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="cue-data")
    sub = parser.add_subparsers(dest="command", required=True)

    pull = sub.add_parser("pull", help="Download and rehydrate published annotations")
    pull.add_argument("--repo-id", default="handshake-ai-research/cue-annotations")
    pull.add_argument("--out-root", default=None)
    pull.add_argument("--config-name", default=None)
    pull.add_argument("--split", action="append", choices=("train", "validation"))
    pull.add_argument("--revision", default=None)
    pull.add_argument("--token", default=None)
    pull.add_argument("--max-records-per-source", type=int, default=0)
    pull.add_argument(
        "--min-coverage",
        type=float,
        default=0.95,
        help="Fail before writing if any source hydrates below this fraction.",
    )
    pull.add_argument(
        "--resume",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Skip records already present under --out-root (default: on).",
    )

    preprocess = sub.add_parser("preprocess", help="Normalize a raw corpus")
    preprocess.add_argument("source", choices=("dialogstudio", "lmsys", "wildchat", "ms-tod", "locomo"))
    preprocess.add_argument("--input", default=None)
    preprocess.add_argument("--out-dir", required=True)
    preprocess.add_argument("--split", default="train")
    preprocess.add_argument("--dataset", default=None)
    preprocess.add_argument("--repo", default=None)
    preprocess.add_argument("--max-records", type=int, default=0)

    audit = sub.add_parser("audit", help="Diff local annotated turns against their upstream source")
    audit.add_argument("--annotated-root", required=True)
    audit.add_argument("--split", default="train")
    audit.add_argument("--dataset", action="append", default=None)
    audit.add_argument("--sample-per-dataset", type=int, default=25)
    audit.add_argument("--out", default=None, help="Optional JSON report path.")

    annotate = sub.add_parser("annotate", help="Build dual-contrast training annotations")
    annotate.add_argument("--config", required=True)
    annotate.add_argument("--data-root", default=None)
    annotate.add_argument("--output-root", default=None)

    args = parser.parse_args(argv)
    if args.command == "pull":
        from cue_training.preprocessing.hydrate_annotations import hydrate_annotations

        hydrate_annotations(
            repo_id=args.repo_id,
            out_root=args.out_root,
            config_name=args.config_name,
            splits=args.split,
            revision=args.revision,
            token=args.token,
            max_records_per_source=args.max_records_per_source or None,
            min_coverage=args.min_coverage,
            resume=args.resume,
        )
    elif args.command == "preprocess":
        from cue_training.preprocessing.raw.__main__ import main as raw_main

        raw_args = [args.source, "--out-dir", args.out_dir, "--split", args.split]
        for name, flag in (("input", "--input"), ("dataset", "--dataset"), ("repo", "--repo")):
            if getattr(args, name):
                raw_args += [flag, getattr(args, name)]
        if args.max_records:
            raw_args += ["--max-records", str(args.max_records)]
        raw_main(raw_args)
    elif args.command == "audit":
        import json
        from cue_training.data.annotation_release import audit_annotated_records

        report = audit_annotated_records(
            args.annotated_root, split=args.split, datasets=args.dataset, sample_per_dataset=args.sample_per_dataset
        )
        text = json.dumps(report, ensure_ascii=False, indent=2)
        if args.out:
            from pathlib import Path

            Path(args.out).write_text(text, encoding="utf-8")
        else:
            print(text)
    elif args.command == "annotate":
        from cue_training.preprocessing.data_annotation import run_data_annotation
        from cue_training.utils.config import load_config

        cfg = load_config(args.config)
        if args.data_root:
            cfg["data_root"] = args.data_root
        if args.output_root:
            cfg["output_root"] = args.output_root
        run_data_annotation(cfg)
