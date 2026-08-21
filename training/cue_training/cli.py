"""cue CLI."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

from cue_training.utils.log import log

TAG = "cli"


def _visible_cuda_devices(requested: int = 0) -> list[str]:
    """Return device identifiers for automatic one-process-per-GPU launch."""

    visible = os.environ.get("CUDA_VISIBLE_DEVICES")
    if visible is not None:
        devices = [item.strip() for item in visible.split(",") if item.strip() and item.strip() != "-1"]
    else:
        import torch

        devices = [str(i) for i in range(torch.cuda.device_count())]
    if requested < 0:
        raise ValueError("--gpus must be non-negative")
    count = len(devices) if requested == 0 else requested
    if count > len(devices):
        raise ValueError(f"--gpus={count}, but only {len(devices)} CUDA devices are visible")
    return devices[:count] if count > 0 else [""]


def main(argv: list[str] | None = None) -> None:
    argv = list(sys.argv[1:] if argv is None else argv)
    if argv in (["model"], ["model", "--help"], ["model", "-h"]):
        print("usage: cue model {train,refine,encode,generate} [options]")
        return
    if argv in (["sampler"], ["sampler", "--help"], ["sampler", "-h"]):
        print("usage: cue sampler {build-bank,train,sample} [options]")
        return
    if len(argv) >= 2 and argv[0] == "model":
        argv = [{"train": "train", "refine": "refine", "encode": "encode", "generate": "generate"}.get(argv[1], argv[1]), *argv[2:]]
    if len(argv) >= 2 and argv[0] == "sampler":
        argv = [{"build-bank": "sampler-build-bank", "train": "sampler-train", "sample": "sampler-sample"}.get(argv[1], argv[1]), *argv[2:]]
    parser = argparse.ArgumentParser(prog="cue-train")
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_train = sub.add_parser("train", help="Joint train encoder+decoder")
    p_train.add_argument("--config", required=True)
    p_train.add_argument("--data_root", default=None)
    p_train.add_argument("--output", default=None)
    p_train.add_argument(
        "--resume",
        default=None,
        help="Checkpoint dir, or auto/last/best (auto prefers last). Fresh optimizer; config LRs apply.",
    )
    p_train.add_argument("--wandb_entity", default=None)
    p_train.add_argument("--wandb_project", default=None)
    p_train.add_argument("--wandb_name", default=None)
    p_train.add_argument("--wandb_group", default=None)
    p_train.add_argument("--no_wandb", action="store_true")

    p_enc = sub.add_parser("encode", help="Encode a session JSON to bottleneck")
    p_enc.add_argument("--checkpoint", required=True)
    p_enc.add_argument("--input", required=True)
    p_enc.add_argument("--device", default="cpu")

    p_export = sub.add_parser("export-embeddings", help="Bulk export evaluation embeddings")
    p_export.add_argument("--checkpoint", required=True)
    p_export.add_argument("--input", required=True)
    p_export.add_argument("--output", required=True)
    p_export.add_argument("--device", default="cpu")
    p_export.add_argument("--batch-size", type=int, default=16)
    p_export.add_argument(
        "--session-preprocess",
        default="full",
        choices=("full", "strip_document", "user_only"),
        help="Preprocess turns before encode (default: full)",
    )

    p_gen = sub.add_parser("generate", help="Generate a persona manual")
    p_gen.add_argument("--checkpoint", required=True)
    p_gen.add_argument("--input", required=True)
    p_gen.add_argument("--device", default="cpu")

    p_s0 = sub.add_parser("annotate", help="Build dual-contrast training set")
    p_s0.add_argument("--config", required=True)
    p_s0.add_argument("--data_root", default=None)
    p_s0.add_argument("--output_root", default=None)

    p_style_aug = sub.add_parser(
        "augment-style-commands",
        help="Add style-kind commands to existing data-annotation manuals without regenerating them",
    )
    p_style_aug.add_argument("--data_root", required=True)
    p_style_aug.add_argument("--output_root", required=True)
    p_style_aug.add_argument("--splits", default="train,validation")
    p_style_aug.add_argument("--model", default="gpt-5.4-mini")
    p_style_aug.add_argument("--n_style", type=int, default=4)
    p_style_aug.add_argument("--api_base", default=None)
    p_style_aug.add_argument("--force", action="store_true")
    p_style_aug.add_argument("--workers", type=int, default=8)

    p_raw = sub.add_parser("raw-preprocess", help="Normalize a raw corpus for data annotation")
    p_raw.add_argument(
        "source", choices=("dialogstudio", "lmsys", "wildchat", "ms-tod", "locomo")
    )
    p_raw.add_argument("--input", default=None)
    p_raw.add_argument("--out-dir", required=True)
    p_raw.add_argument("--split", default="train")
    p_raw.add_argument("--dataset", default=None)
    p_raw.add_argument("--repo", default=None)
    p_raw.add_argument("--max-records", type=int, default=0)

    p_hydrate = sub.add_parser(
        "hydrate-annotations",
        help="Download published annotation deltas and reconstruct training JSONL",
    )
    p_hydrate.add_argument("--repo-id", default="handshake-ai-research/cue-annotations")
    p_hydrate.add_argument("--out-root", default=None)
    p_hydrate.add_argument("--config-name", default=None)
    p_hydrate.add_argument("--split", action="append", choices=("train", "validation"))
    p_hydrate.add_argument("--revision", default=None)
    p_hydrate.add_argument("--token", default=None)
    p_hydrate.add_argument("--max-records-per-source", type=int, default=0)

    p_s2 = sub.add_parser("refine", help="GRPO persona manuals through a user simulator")
    p_s2.add_argument("--config", required=True)
    p_s2.add_argument("--joint-training-checkpoint", default=None)
    p_s2.add_argument("--output", default=None)
    p_s2.add_argument(
        "--nproc",
        type=int,
        default=None,
        help="Override config nproc for torchrun (default: config nproc / len(gpus) / 1).",
    )

    p_bank = sub.add_parser(
        "sampler-build-bank",
        help="Encode prior sessions into the sampler embedding bank + FAISS",
    )
    p_bank.add_argument("--config", required=True)
    p_bank.add_argument("--joint-training-checkpoint", default=None)
    p_bank.add_argument("--bank_dir", default=None)
    p_bank.add_argument("--prior_data_root", default=None)

    p_st = sub.add_parser("sampler-train", help="Train cue embedding diffusion sampler")
    p_st.add_argument("--config", required=True)
    p_st.add_argument("--bank_dir", default=None)
    p_st.add_argument("--output", default=None)
    p_st.add_argument("--joint-training-checkpoint", default=None)
    p_st.add_argument("--resume", default=None)
    p_st.add_argument("--no_wandb", action="store_true")

    p_ss = sub.add_parser("sampler-sample", help="Sample cue embeddings (DDIM + CFG)")
    p_ss.add_argument("--checkpoint", required=True)
    p_ss.add_argument("--n", type=int, default=8)
    p_ss.add_argument("--steps", type=int, default=None)
    p_ss.add_argument("--guidance_w", type=float, default=None)
    p_ss.add_argument(
        "--condition",
        default=None,
        help="JSON/JSONL file of embeddings: [D], [K,D], or {\"embeddings\": [...]}",
    )
    p_ss.add_argument("--output", default=None, help="Write embeddings JSON to this path")
    p_ss.add_argument("--device", default=None)
    p_ss.add_argument("--seed", type=int, default=0)
    p_ss.add_argument("--project_manifold", action="store_true")
    p_ss.add_argument("--reject_decode", action="store_true")
    p_ss.add_argument("--decoder_checkpoint", default=None)

    p_smoke = sub.add_parser("smoke", help="Tiny train/reload/generate smoke test")
    p_smoke.add_argument("--config", default="configs/tiny.json")

    p_style = sub.add_parser(
        "precompute-wegmann-style",
        help="Precompute session-level Wegmann style vectors for joint training distillation",
    )
    p_style.add_argument("--data_root", required=True)
    p_style.add_argument("--output", default=None)
    p_style.add_argument("--split", default="train")
    p_style.add_argument("--model_name", default="AnnaWegmann/Style-Embedding")
    p_style.add_argument("--batch_size", type=int, default=32)
    p_style.add_argument("--session-batch-size", type=int, default=256)
    p_style.add_argument("--max_records", type=int, default=0)
    p_style.add_argument("--shard-index", type=int, default=0)
    p_style.add_argument("--num-shards", type=int, default=1)
    p_style.add_argument(
        "--gpus",
        type=int,
        default=0,
        help="GPU workers (0 = all visible GPUs; 1 = single process)",
    )
    p_style.add_argument("--keep-shards", action="store_true")

    p_merge = sub.add_parser(
        "merge-wegmann-style",
        help="Merge sharded Wegmann style JSONLs into one sidecar",
    )
    p_merge.add_argument("--shards", nargs="+", required=True)
    p_merge.add_argument("--output", required=True)

    p_pool = sub.add_parser(
        "build-example-pool",
        help="Build CUE-keyed example pool for inference retrieval",
    )
    p_pool.add_argument("--checkpoint", required=True)
    p_pool.add_argument("--data_root", required=True)
    p_pool.add_argument("--output", default=None)
    p_pool.add_argument("--split", default="train")
    p_pool.add_argument("--batch_size", type=int, default=64,
                        help="Sessions per encode pass. Layerwise pooling stacks [L,turns,T,H], so "
                             "the peak allocation scales with this; lower it if a rank OOMs.")
    p_pool.add_argument("--device", default="cpu",
                        help="Ignored under torchrun; each rank uses its own LOCAL_RANK GPU.")
    p_pool.add_argument("--max_records", type=int, default=0)

    p_hf = sub.add_parser(
        "export-cue-hf",
        help="Export a checkpoint for the sibling cue-hf inference package",
    )
    p_hf.add_argument("--checkpoint", required=True)
    p_hf.add_argument("--output", required=True)
    p_hf.add_argument("--device", default="cpu")
    p_hf.add_argument("--sampler_checkpoint", default=None,
                      help="Copy this sampler.pt into the exported model dir")
    p_hf.add_argument("--sampler_id", default=None,
                      help="Hub repo cue-hf should pull sampler.pt from instead")
    p_hf.add_argument("--example_pool_id", default=None,
                      help="Hub dataset id of the example pool (see export-example-pool-dataset)")
    p_hf.add_argument(
        "--session_preprocess",
        default=None,
        choices=("full", "strip_document", "user_only"),
        help="Pin a non-default encode preprocess in the exported config (default: full)",
    )
    p_hf.add_argument("--push_to", default=None, help="Hub model repo id to upload to")
    p_hf.add_argument("--public", action="store_true")

    p_hf_pool = sub.add_parser(
        "export-example-pool-dataset",
        help="Export a built example pool as a HF dataset for cue-hf retrieval",
    )
    p_hf_pool.add_argument("--checkpoint", required=True)
    p_hf_pool.add_argument("--output", default=None, help="save_to_disk target")
    p_hf_pool.add_argument("--push_to", default=None, help="Hub dataset repo id")
    p_hf_pool.add_argument("--public", action="store_true")

    args = parser.parse_args(argv)

    if args.cmd == "train":
        from cue_training.training.train import train_from_config
        from cue_training.utils.config import load_config

        cfg = load_config(args.config)
        if args.data_root:
            cfg["data_root"] = args.data_root
        if args.output:
            cfg["output"] = args.output
        if args.resume is not None:
            cfg["resume"] = args.resume
        if args.wandb_entity is not None:
            cfg["wandb_entity"] = args.wandb_entity
        if args.wandb_project is not None:
            cfg["wandb_project"] = args.wandb_project
        if args.wandb_name is not None:
            cfg["wandb_name"] = args.wandb_name
        if args.wandb_group is not None:
            cfg["wandb_group"] = args.wandb_group
        if args.no_wandb:
            cfg["no_wandb"] = True
        train_from_config(cfg)
        return

    if args.cmd == "encode":
        from cue_training.infer import encode_file

        emb = encode_file(args.checkpoint, args.input, device=args.device)
        print(json.dumps({"embedding": emb}))
        return

    if args.cmd == "export-embeddings":
        from cue_training.infer.export import export_embeddings

        export_embeddings(
            args.checkpoint,
            args.input,
            args.output,
            device=args.device,
            batch_size=args.batch_size,
            session_preprocess=args.session_preprocess,
        )
        return

    if args.cmd == "generate":
        from cue_training.infer import generate_file

        manual = generate_file(args.checkpoint, args.input, device=args.device)
        print(json.dumps(manual, ensure_ascii=False, indent=2))
        return

    if args.cmd == "annotate":
        from cue_training.preprocessing.data_annotation import run_data_annotation
        from cue_training.utils.config import load_config

        cfg = load_config(args.config)
        if args.data_root:
            cfg["data_root"] = args.data_root
        if args.output_root:
            cfg["output_root"] = args.output_root
        run_data_annotation(cfg)
        return

    if args.cmd == "augment-style-commands":
        from cue_training.preprocessing.style_augment import run_style_augment

        run_style_augment(
            data_root=args.data_root,
            output_root=args.output_root,
            splits=[s.strip() for s in str(args.splits).split(",") if s.strip()],
            model=args.model,
            n_style=max(1, int(args.n_style)),
            api_base=args.api_base,
            force=bool(args.force),
            workers=max(1, int(args.workers)),
        )
        return

    if args.cmd == "raw-preprocess":
        from cue_training.preprocessing.raw.__main__ import main as raw_main

        raw_args = [args.source, "--out-dir", args.out_dir, "--split", args.split]
        if args.input:
            raw_args.extend(["--input", args.input])
        if args.dataset:
            raw_args.extend(["--dataset", args.dataset])
        if args.repo:
            raw_args.extend(["--repo", args.repo])
        if args.max_records:
            raw_args.extend(["--max-records", str(args.max_records)])
        raw_main(raw_args)
        return

    if args.cmd == "hydrate-annotations":
        from cue_training.preprocessing.hydrate_annotations import hydrate_annotations

        hydrate_annotations(
            repo_id=args.repo_id,
            out_root=args.out_root,
            config_name=args.config_name,
            splits=args.split,
            revision=args.revision,
            token=args.token,
            max_records_per_source=(
                args.max_records_per_source if args.max_records_per_source > 0 else None
            ),
        )
        return

    if args.cmd == "refine":
        from cue_training.utils.config import load_config

        cfg = load_config(args.config)
        if args.joint_training_checkpoint:
            cfg["joint_training_checkpoint"] = args.joint_training_checkpoint
        if args.output:
            cfg["output"] = args.output

        # Auto-launch multi-GPU training via torchrun when not already under a launcher.
        world_size = int(os.environ.get("WORLD_SIZE", "1"))
        gpus = [int(g) for g in (cfg.get("gpus") or [])]
        nproc = args.nproc
        if nproc is None:
            nproc = int(cfg.get("nproc") or (len(gpus) if gpus else 1))
        if nproc > 1 and world_size == 1:
            if gpus:
                if len(gpus) < nproc:
                    raise SystemExit(f"refine nproc={nproc} but only {len(gpus)} gpus listed")
                visible = ",".join(str(g) for g in gpus[:nproc])
            else:
                visible = os.environ.get("CUDA_VISIBLE_DEVICES", "")
                if not visible:
                    visible = ",".join(str(i) for i in range(nproc))
            cmd = [
                sys.executable,
                "-m",
                "torch.distributed.run",
                f"--nproc_per_node={nproc}",
                "-m",
                "cue_training.cli",
                "refine",
                "--config",
                str(args.config),
            ]
            if args.joint_training_checkpoint:
                cmd += ["--joint-training-checkpoint", str(args.joint_training_checkpoint)]
            if args.output:
                cmd += ["--output", str(args.output)]
            env = os.environ.copy()
            env["CUDA_VISIBLE_DEVICES"] = visible
            log(TAG, f"launching refine torchrun nproc={nproc} CUDA_VISIBLE_DEVICES={visible}")
            raise SystemExit(subprocess.call(cmd, env=env))

        # Single-process: `gpus` used to be honored only on the torchrun path above, so
        # `device: cuda` resolved to physical GPU 0 and collided with whatever was there.
        # Set this before importing the trainer so torch never initializes on cuda:0.
        if gpus and world_size == 1 and not os.environ.get("CUDA_VISIBLE_DEVICES"):
            os.environ["CUDA_VISIBLE_DEVICES"] = str(gpus[0])
            log(TAG, f"refine pinned to GPU {gpus[0]} (CUDA_VISIBLE_DEVICES)")

        from cue_training.refinement.train import train_refinement_from_config

        train_refinement_from_config(cfg)
        return

    if args.cmd == "sampler-build-bank":
        from cue_training.sampler.bank import build_bank_distributed
        from cue_training.utils.config import load_config

        cfg = load_config(args.config)
        if args.joint_training_checkpoint:
            cfg["joint_training_checkpoint"] = args.joint_training_checkpoint
        if args.bank_dir:
            cfg["bank_dir"] = args.bank_dir
        if args.prior_data_root:
            cfg["prior_data_root"] = args.prior_data_root
        bank = build_bank_distributed(cfg)
        if bank is not None:  # non-zero torchrun ranks return None after their shard
            log(TAG, f"bank at {bank.root} counts={bank.meta and bank.meta.get('counts')}")
        return

    if args.cmd == "sampler-train":
        from cue_training.sampler.bank import build_bank_distributed, load_bank
        from cue_training.sampler.train import train_sampler_from_config
        from cue_training.utils.config import load_config

        cfg = load_config(args.config)
        if args.bank_dir:
            cfg["bank_dir"] = args.bank_dir
        if args.output:
            cfg["output"] = args.output
        if args.joint_training_checkpoint:
            cfg["joint_training_checkpoint"] = args.joint_training_checkpoint
        if args.resume:
            cfg["resume"] = args.resume
        if args.no_wandb:
            cfg["no_wandb"] = True
        bank_dir = Path(cfg.get("bank_dir") or cfg.get("output") or "")
        try:
            load_bank(bank_dir)
        except (FileNotFoundError, ValueError):
            log(TAG, f"sampler bank unavailable at {bank_dir}; building it first")
            build_bank_distributed(cfg)
        out = train_sampler_from_config(cfg)
        log(TAG, f"sampler training finished -> {out}")
        return

    if args.cmd == "sampler-sample":
        from cue_training.sampler.sample import sample

        condition = None
        if args.condition:
            condition = _load_condition_embeddings(args.condition)
        result = sample(
            args.checkpoint,
            n=args.n,
            steps=args.steps,
            guidance_w=args.guidance_w,
            condition_embeddings=condition,
            device=args.device,
            project_manifold=True if args.project_manifold else None,
            reject_decode=args.reject_decode,
            decoder_checkpoint=args.decoder_checkpoint,
            seed=args.seed,
        )
        payload = {
            "embeddings": result["embeddings"].tolist(),
            "config": result["config"],
            "attempts": result["attempts"],
        }
        if args.output:
            Path(args.output).write_text(
                json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
            )
            log(TAG, f"wrote {args.n} embeddings -> {args.output}")
        else:
            print(json.dumps(payload, ensure_ascii=False))
        return

    if args.cmd == "smoke":
        _smoke(args.config)
        return

    if args.cmd == "precompute-wegmann-style":
        from cue_training.data.wegmann_style import (
            build_wegmann_sidecar,
            merge_wegmann_shards,
            sidecar_path,
        )

        output = args.output or sidecar_path(args.data_root, split=args.split)
        devices = _visible_cuda_devices(args.gpus)
        if len(devices) > 1 and int(args.num_shards) == 1:
            output = Path(output)
            shard_paths = [
                output.with_name(
                    f"{output.stem}.shard{i:02d}-of-{len(devices):02d}{output.suffix}"
                )
                for i in range(len(devices))
            ]
            processes: list[subprocess.Popen] = []
            for i, (device, shard_path) in enumerate(zip(devices, shard_paths, strict=True)):
                command = [
                    sys.executable,
                    "-m",
                    "cue_training.cli",
                    "precompute-wegmann-style",
                    "--data_root",
                    str(args.data_root),
                    "--output",
                    str(shard_path),
                    "--split",
                    str(args.split),
                    "--model_name",
                    str(args.model_name),
                    "--batch_size",
                    str(args.batch_size),
                    "--session-batch-size",
                    str(args.session_batch_size),
                    "--shard-index",
                    str(i),
                    "--num-shards",
                    str(len(devices)),
                    "--gpus",
                    "1",
                ]
                if args.max_records:
                    command.extend(["--max_records", str(args.max_records)])
                env = os.environ.copy()
                env["CUDA_VISIBLE_DEVICES"] = device
                log(TAG, f"launching Wegmann shard {i + 1}/{len(devices)} on GPU {device}")
                processes.append(subprocess.Popen(command, env=env))
            failed = []
            for i, process in enumerate(processes):
                code = process.wait()
                if code:
                    failed.append((i, code))
            if failed:
                raise RuntimeError(f"Wegmann GPU workers failed: {failed}")
            merge_wegmann_shards(shard_paths, output)
            if not args.keep_shards:
                for shard_path in shard_paths:
                    shard_path.unlink(missing_ok=True)
            return
        if int(args.num_shards) > 1 and args.output is None:
            output = Path(str(output)).with_name(
                f"wegmann_style.shard{int(args.shard_index):02d}-of-{int(args.num_shards):02d}.jsonl"
            )
        build_wegmann_sidecar(
            args.data_root,
            output,
            split=args.split,
            model_name=args.model_name,
            batch_size=args.batch_size,
            session_batch_size=args.session_batch_size,
            max_records=args.max_records,
            shard_index=int(args.shard_index),
            num_shards=int(args.num_shards),
        )
        return

    if args.cmd == "merge-wegmann-style":
        from cue_training.data.wegmann_style import merge_wegmann_shards

        merge_wegmann_shards(args.shards, args.output)
        return

    if args.cmd == "build-example-pool":
        from cue_training.data.example_pool import (
            build_example_pool_distributed,
            pool_dir_for_checkpoint,
        )

        output = args.output or pool_dir_for_checkpoint(args.checkpoint)
        build_example_pool_distributed(
            args.checkpoint,
            args.data_root,
            output,
            split=args.split,
            batch_size=args.batch_size,
            device=args.device,
            max_records=args.max_records,
        )
        return

    if args.cmd == "export-cue-hf":
        from cue_training.infer.export_cue_hf import export_cue_hf

        export_cue_hf(
            args.checkpoint,
            args.output,
            device=args.device,
            sampler_checkpoint=args.sampler_checkpoint,
            sampler_id=args.sampler_id,
            example_pool_id=args.example_pool_id,
            session_preprocess=args.session_preprocess,
            push_to=args.push_to,
            private=not args.public,
        )
        return

    if args.cmd == "export-example-pool-dataset":
        from cue_training.infer.export_cue_hf import export_example_pool_dataset

        export_example_pool_dataset(
            args.checkpoint,
            output=args.output,
            push_to=args.push_to,
            private=not args.public,
        )
        return


def _load_condition_embeddings(path: str):
    import numpy as np

    text = Path(path).read_text(encoding="utf-8").strip()
    if path.endswith(".jsonl"):
        rows = [json.loads(line) for line in text.splitlines() if line.strip()]
        vecs = []
        for row in rows:
            if isinstance(row, list):
                vecs.append(row)
            elif isinstance(row, dict):
                vecs.append(row.get("embedding") or row.get("trajectory_embedding") or row["embeddings"])
            else:
                raise ValueError(f"bad condition row type: {type(row)}")
        return np.asarray(vecs, dtype=np.float32)
    data = json.loads(text)
    if isinstance(data, dict):
        data = data.get("embeddings") or data.get("embedding") or data.get("trajectory_embedding")
    return np.asarray(data, dtype=np.float32)


def _smoke(config_path: str) -> None:
    """Create a tiny fixture, optionally skip heavy model download if unavailable."""

    from cue_training.data.schema import validate_manual
    from cue_training.data.streaming import write_jsonl_atomic

    root = Path(tempfile.mkdtemp(prefix="cue_smoke_"))
    data = root / "toy" / "train.jsonl"
    data.parent.mkdir(parents=True)
    manual = validate_manual(
        {
            "commands": [
                {"text": "Be terse and skip pleasantries.", "kind": "sim_contrast", "form": "general", "examples": ["ok"]},
                {"text": "Push back when misunderstood.", "kind": "sim_contrast", "form": "if_then", "examples": ["thats not what i asked"]},
                {"text": "Use lowercase and few commas.", "kind": "human_contrast", "form": "general", "examples": ["fix it"]},
                {"text": "Ask short clarifying questions.", "kind": "human_contrast", "form": "general", "examples": ["which one?"]},
                {"text": "Avoid markdown.", "kind": "sim_contrast", "form": "general", "examples": ["send the file"]},
                {"text": "If the agent apologizes, ignore it and restate the ask.", "kind": "sim_contrast", "form": "if_then", "examples": ["just do it"]},
                {"text": "Prefer fragments over full sentences.", "kind": "human_contrast", "form": "general", "examples": ["still broken"]},
                {"text": "If asked for details, give one line only.", "kind": "human_contrast", "form": "if_then", "examples": ["timeout on login"]},
                {"text": "Do not thank the assistant.", "kind": "sim_contrast", "form": "general", "examples": ["next"]},
                {"text": "Stay impatient across turns.", "kind": "human_contrast", "form": "general", "examples": ["hurry"]},
            ]
        }
    )
    write_jsonl_atomic(
        data,
        [
            {
                "id": "s1",
                "dataset": "toy",
                "split": "train",
                "turns": [
                    {"role": "assistant", "content": "How can I help?"},
                    {"role": "user", "content": "fix login timeout"},
                    {"role": "assistant", "content": "Can you share logs?"},
                    {"role": "user", "content": "still broken"},
                ],
                "persona_manual": manual,
            }
        ],
    )
    log(TAG, f"smoke fixture at {data}")
    # Schema/streaming smoke always runs; full train requires HF models.
    from cue_training.data.streaming import RoundRobinStreamingDataset

    ds = RoundRobinStreamingDataset(root, split="train", per_source_cap=10, shuffle_buffer=0)
    n = sum(1 for _ in ds)
    assert n == 1, n
    log(TAG, "streaming smoke OK")
    cfg_path = Path(config_path)
    if not cfg_path.exists():
        # Resolve relative to package root.
        cfg_path = Path(__file__).resolve().parents[1] / config_path
    if cfg_path.exists():
        from cue_training.utils.config import load_config
        from cue_training.training.train import train_from_config

        cfg = load_config(cfg_path)
        cfg["data_root"] = str(root)
        cfg["output"] = str(root / "ckpt")
        cfg["max_steps"] = int(cfg.get("max_steps", 1))
        cfg["epochs"] = 1
        try:
            train_from_config(cfg)
            log(TAG, "train smoke OK")
        except Exception as exc:  # noqa: BLE001
            log(TAG, f"train smoke skipped/failed (likely missing HF weights): {exc}")
    else:
        log(TAG, f"no config at {cfg_path}; schema/streaming smoke only")


if __name__ == "__main__":
    main()
