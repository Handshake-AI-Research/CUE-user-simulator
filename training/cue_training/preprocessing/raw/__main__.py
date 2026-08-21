"""CLI for local, network-free raw-corpus conversion."""

from __future__ import annotations

import argparse
from collections.abc import Mapping

from cue_training.preprocessing.raw import (
    process_dialoguestudio,
    process_lmsys,
    process_locomo,
    process_ms_tod,
    process_wildchat,
)
from cue_training.preprocessing.raw.utils import load_records, write_dataset_split


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Normalize raw conversations for data annotation.")
    parser.add_argument("source", choices=("dialogstudio", "lmsys", "wildchat", "ms-tod", "locomo"))
    parser.add_argument("--input", default=None, help="Local JSON/JSONL; omit for supported HF sources.")
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--split", default="train")
    parser.add_argument("--dataset", default=None, help="Required DialogStudio config name.")
    parser.add_argument("--repo", default=None, help="Optional Hugging Face repository override.")
    parser.add_argument("--max-records", type=int, default=0)
    args = parser.parse_args(argv)

    if args.input:
        data = load_records(args.input)
    elif args.source in {"dialogstudio", "lmsys", "wildchat"}:
        from datasets import load_dataset

        defaults = {
            "dialogstudio": "Salesforce/dialogstudio",
            "lmsys": "lmsys/lmsys-chat-1m",
            "wildchat": "allenai/WildChat-1M",
        }
        repo = args.repo or defaults[args.source]
        if args.source == "dialogstudio":
            if not args.dataset:
                parser.error("--dataset is required for dialogstudio")
            try:
                loaded = load_dataset(repo, args.dataset, trust_remote_code=True)
            except TypeError:
                loaded = load_dataset(repo, args.dataset)
            data = loaded[args.split]
        else:
            data = load_dataset(repo, split=args.split)
    else:
        parser.error("--input is required for ms-tod and locomo")
    if args.source == "dialogstudio":
        if not args.dataset:
            parser.error("--dataset is required for dialogstudio")
        records = process_dialoguestudio.convert_rows(data, args.dataset, split=args.split)
        dataset = args.dataset
    elif args.source == "lmsys":
        records = process_lmsys.convert_rows(data, split=args.split)
        dataset = process_lmsys.DATASET_NAME
    elif args.source == "wildchat":
        records = process_wildchat.convert_rows(data, split=args.split)
        dataset = process_wildchat.DATASET_NAME
    elif args.source == "ms-tod":
        records = process_ms_tod.convert_dataset(data, split=args.split)
        dataset = process_ms_tod.DATASET_NAME
    else:
        if isinstance(data, Mapping):
            data = data.get("data", data.get("samples", [data]))
        records = process_locomo.convert_dataset(data, split=args.split)
        dataset = process_locomo.DATASET_NAME

    if args.max_records > 0:
        records = records[: args.max_records]
    path = write_dataset_split(records, args.out_dir, dataset, args.split)
    print(f"{dataset}/{args.split}: {len(records)} records -> {path}")


if __name__ == "__main__":
    main()
