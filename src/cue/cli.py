"""CLI: condition on a conversation, sample users, or run one steered turn."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any

from cue.conversation import load_conversation
from cue.manual import PersonaManual
from cue.runtime import DEFAULT_REPO, Cue
from cue.simulator import (
    HuggingFaceSimulator,
    OpenAICompatSimulator,
    run_user_turn,
)

_SOURCE_REQUIRED = "pass --conversation, --sample, or --manual"


def _add_model_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--model",
        default=os.environ.get("CUE_REPO", DEFAULT_REPO),
        help=f"Hub repo or local export (default: {DEFAULT_REPO})",
    )
    parser.add_argument("--device", default=os.environ.get("CUE_DEVICE", "cpu"))
    parser.add_argument(
        "--token",
        default=os.environ.get("HF_TOKEN"),
        help="Hugging Face token (or set HF_TOKEN)",
    )
    parser.add_argument(
        "--example-pool",
        default=None,
        help="Optional Hub dataset id for style-sample retrieval",
    )
    parser.add_argument(
        "--session-preprocess",
        default="full",
        choices=("full", "strip_document", "user_only"),
    )


def _load_cue(args: argparse.Namespace) -> Cue:
    return Cue.from_pretrained(
        args.model,
        device=args.device,
        token=args.token,
        example_pool=args.example_pool,
        session_preprocess=args.session_preprocess,
    )


def _print_manual(manual: Any, *, as_json: bool) -> None:
    if as_json:
        sys.stdout.write(json.dumps(manual.to_dict(), indent=2) + "\n")
        return
    sys.stdout.write((manual.steering_prompt or "(empty manual)") + "\n")


def cmd_conditioned(args: argparse.Namespace) -> int:
    cue = _load_cue(args)
    messages: list[dict[str, str]] | str
    if args.conversation:
        messages = load_conversation(args.conversation)
    elif args.text:
        messages = args.text
    else:
        messages = sys.stdin.read()
    manual = cue.from_conversation(
        messages,
        session_preprocess=args.session_preprocess,
        example_retrieval=bool(args.retrieve),
    )
    _print_manual(manual, as_json=args.json)
    return 0


def cmd_sample(args: argparse.Namespace) -> int:
    cue = _load_cue(args)
    out = cue.sample(
        n=args.n,
        seed=args.seed,
        example_retrieval=bool(args.retrieve),
    )
    manuals = out if isinstance(out, list) else [out]
    if args.json:
        sys.stdout.write(
            json.dumps(
                [
                    {
                        "source": m.source,
                        "commands": m.commands,
                        "examples": m.examples,
                        "steering_prompt": m.steering_prompt,
                    }
                    for m in manuals
                ],
                indent=2,
            )
            + "\n"
        )
    else:
        for i, manual in enumerate(manuals, start=1):
            if len(manuals) > 1:
                sys.stdout.write(f"=== user {i} ===\n")
            sys.stdout.write((manual.steering_prompt or "(empty manual)") + "\n")
            if i < len(manuals):
                sys.stdout.write("\n")
    return 0


def _build_simulator(args: argparse.Namespace) -> Any:
    backend = args.simulator
    if backend == "openai":
        return OpenAICompatSimulator(args.sim_model, api_key=args.api_key, base_url=args.base_url)
    if backend == "hf":
        return HuggingFaceSimulator(args.sim_model, token=args.token)
    message = f"unknown simulator backend: {backend}"
    raise SystemExit(message)


def cmd_chat(args: argparse.Namespace) -> int:
    cue = _load_cue(args)
    if args.conversation:
        manual = cue.from_conversation(
            load_conversation(args.conversation),
            session_preprocess=args.session_preprocess,
            example_retrieval=bool(args.retrieve),
        )
    elif args.sample:
        manual = cue.sample(seed=args.seed)  # type: ignore[assignment]
    elif args.manual:
        text = Path(args.manual).read_text(encoding="utf-8")
        manual = PersonaManual(commands=[], steering_prompt=text.strip())
    else:
        raise SystemExit(_SOURCE_REQUIRED)

    sim = _build_simulator(args)
    history: list[dict[str, str]] = []
    if args.history:
        history = load_conversation(args.history)
    user_msg = run_user_turn(sim, scenario=args.scenario, history=history, manual=manual)
    sys.stdout.write(user_msg + "\n")
    return 0


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        prog="cue",
        description="Production CUE runtime: condition, sample, or steer a simulator",
    )
    parser.add_argument(
        "--config",
        type=Path,
        help="Run from a TOML config (the recommended production interface)",
    )
    sub = parser.add_subparsers(dest="cmd")

    conditioned = sub.add_parser("conditioned", help="Decode a user-conditioned manual from a conversation")
    _add_model_args(conditioned)
    conditioned.add_argument("-c", "--conversation", help="JSON / JSONL / transcript file")
    conditioned.add_argument("-t", "--text", help="Inline transcript string")
    conditioned.add_argument("--retrieve", action="store_true")
    conditioned.add_argument("--json", action="store_true")
    conditioned.set_defaults(func=cmd_conditioned)

    sample = sub.add_parser("sample", help="Sample synthetic users from the prior")
    _add_model_args(sample)
    sample.add_argument("-n", type=int, default=1)
    sample.add_argument("--seed", type=int, default=0)
    sample.add_argument("--retrieve", action="store_true")
    sample.add_argument("--json", action="store_true")
    sample.set_defaults(func=cmd_sample)

    chat = sub.add_parser("chat", help="Run one steered user turn against a base simulator")
    _add_model_args(chat)
    chat.add_argument("--scenario", required=True, help="Task the simulated user is trying to do")
    src = chat.add_mutually_exclusive_group(required=True)
    src.add_argument("-c", "--conversation", help="Condition on this conversation")
    src.add_argument("--sample", action="store_true", help="Sample a fresh user instead")
    src.add_argument("--manual", help="Use a pre-rendered steering prompt file")
    chat.add_argument("--history", help="Optional prior dialogue JSON/transcript")
    chat.add_argument("--seed", type=int, default=0)
    chat.add_argument("--retrieve", action="store_true")
    chat.add_argument(
        "--simulator",
        choices=("openai", "hf"),
        default="openai",
        help="Backend for the base simulator LM",
    )
    chat.add_argument(
        "--sim-model",
        default=os.environ.get("SIM_MODEL", "gpt-4o-mini"),
        help="Model id for the simulator backend",
    )
    chat.add_argument("--api-key", default=None)
    chat.add_argument("--base-url", default=None)
    chat.set_defaults(func=cmd_chat)

    args = parser.parse_args(argv)
    if args.config:
        if args.cmd:
            parser.error("--config cannot be combined with a subcommand")
        from cue.config import CueConfig  # noqa: PLC0415
        from cue.runner import run  # noqa: PLC0415

        result = run(CueConfig.from_toml(args.config), token=os.environ.get("HF_TOKEN"))
        sys.stdout.write(json.dumps(result, indent=2) + "\n")
        return
    if not args.cmd:
        parser.error("pass --config or a subcommand")
    raise SystemExit(args.func(args))


if __name__ == "__main__":
    main()
