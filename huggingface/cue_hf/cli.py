"""cue-hf command line: decode manuals from sessions, or sample synthetic users."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

from cue_hf.modeling_cue import CueModel
from cue_hf.processing_cue import CueProcessor
from cue_hf.render import render_manual


def _load_sessions(path: str | Path) -> list[list[dict[str, str]]]:
    """Read a session JSON ({turns: [...]} or a message list) or JSONL of sessions."""

    path = Path(path)
    processor = CueProcessor()
    if path.suffix == ".jsonl":
        rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    else:
        rows = [json.loads(path.read_text(encoding="utf-8"))]
    sessions = []
    for row in rows:
        messages = row.get("turns") or row.get("messages") if isinstance(row, dict) else row
        sessions.extend(processor(messages))
    return sessions


def _load_condition(path: str | Path | None) -> np.ndarray | None:
    if path is None:
        return None
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if isinstance(payload, dict):
        payload = payload.get("embedding") or payload.get("trajectory_embedding")
    return np.asarray(payload, dtype=np.float32)


def _emit(manuals: list[dict[str, Any] | None], output: str | None, *, render: bool) -> None:
    lines = [json.dumps({"manual": manual, "render": render_manual(manual) if render else None}) for manual in manuals]
    text = "\n".join(lines) + "\n"
    if output:
        Path(output).write_text(text, encoding="utf-8")
    else:
        print(text, end="")


def _slim_sampler(args: Any) -> int:
    import tempfile

    from cue_hf.sampler.sample import SAMPLER_NAME, slim_sampler_file

    source = Path(args.sampler)
    if not source.exists():
        from huggingface_hub import hf_hub_download

        source = Path(hf_hub_download(str(args.sampler), SAMPLER_NAME))
    output = Path(args.output or Path(tempfile.mkdtemp()) / SAMPLER_NAME)
    before = (source if source.is_file() else source / SAMPLER_NAME).stat().st_size
    slim_sampler_file(source, output)
    after = output.stat().st_size
    print(f"{before:,} -> {after:,} bytes ({before / max(after, 1):.1f}x smaller): {output}")
    if args.push_to:
        from huggingface_hub import HfApi

        HfApi().upload_file(path_or_fileobj=str(output), path_in_repo=SAMPLER_NAME, repo_id=args.push_to)
        print(f"replaced {SAMPLER_NAME} in {args.push_to}")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="cue-hf", description=__doc__)
    # Shared flags live on the subcommands so they can be passed after it (git-style).
    shared = argparse.ArgumentParser(add_help=False)
    shared.add_argument("--checkpoint", required=True, help="cue-hf repo id or checkpoint dir")
    shared.add_argument("--device", default="cpu")
    shared.add_argument("--sampler", default=None, help="sampler dir/repo holding sampler.pt")
    shared.add_argument("--example-pool", default=None, help="pool dir or Hub dataset id")
    shared.add_argument("--example-retrieval", action="store_true")
    shared.add_argument("--render", action="store_true", help="also emit simulator text")
    shared.add_argument("--output", default=None)
    sub = parser.add_subparsers(dest="command", required=True)

    gen = sub.add_parser("generate", parents=[shared], help="decode manuals for real sessions")
    gen.add_argument("--input", required=True, help="session JSON or JSONL")
    gen.add_argument(
        "--session-preprocess",
        default=None,
        choices=("full", "strip_document", "user_only"),
        help="encode-time turn preprocess (eval uses strip_document for writing)",
    )

    smp = sub.add_parser("sample", parents=[shared], help="sample synthetic users and decode their manuals")
    smp.add_argument("--n", type=int, default=1)
    smp.add_argument("--condition", default=None, help="JSON [D] or [K,D] condition vectors")
    smp.add_argument("--guidance-w", type=float, default=None)
    smp.add_argument("--steps", type=int, default=None)
    smp.add_argument("--seed", type=int, default=0)

    bundle = sub.add_parser(
        "bundle-remote-code",
        help="add this package to an exported model dir so it loads without installing",
    )
    target = bundle.add_mutually_exclusive_group(required=True)
    target.add_argument("--directory", help="exported cue-hf model dir")
    target.add_argument("--repo", help="already-pushed repo id (weights are not downloaded)")
    bundle.add_argument("--push_to", default=None, help="Hub model repo id to upload to")
    bundle.add_argument("--public", action="store_true")

    slim = sub.add_parser(
        "slim-sampler",
        help="strip optimizer/EMA training state from a sampler.pt (4x smaller)",
    )
    slim.add_argument("--sampler", required=True, help="sampler.pt, its dir, or a repo id")
    slim.add_argument("--output", default=None, help="write here instead of a temp file")
    slim.add_argument("--push_to", default=None, help="replace sampler.pt in this repo")

    args = parser.parse_args(argv)
    if args.command == "slim-sampler":
        return _slim_sampler(args)
    if args.command == "bundle-remote-code":
        from cue_hf.remote_code import bundle_remote_code, bundle_remote_code_in_repo

        if args.repo:
            bundle_remote_code_in_repo(args.repo, private=not args.public)
        else:
            bundle_remote_code(args.directory, push_to=args.push_to, private=not args.public)
        return 0
    model = CueModel.from_pretrained(
        args.checkpoint,
        device=args.device,
        sampler_checkpoint=args.sampler,
        example_pool=args.example_pool,
    )
    if args.command == "generate":
        manuals = model.generate_manual(
            sessions=_load_sessions(args.input),
            example_retrieval=args.example_retrieval,
            session_preprocess=args.session_preprocess,
        )
    else:
        out = model.sample_user(
            n=args.n,
            condition=_load_condition(args.condition),
            guidance_w=args.guidance_w,
            steps=args.steps,
            seed=args.seed,
            example_retrieval=args.example_retrieval,
        )
        manuals = out["manuals"]
    _emit(manuals, args.output, render=args.render)
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
