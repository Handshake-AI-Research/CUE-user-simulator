"""Ray fan-out driver for tau2-bench on Anyscale (CPU head + GPU sidecar).

GPU sidecar actors serve the CUE decoders; one CPU
Ray task per (variant, arm, episode) subprocesses ``tau2-eval run`` for that single
episode with ``--sidecar_url`` (so the tau2 process is CPU-only). Per-episode results +
MirrorBench rollouts are merged at the end.

Run on the Ray head (CUE venv). ``ray`` is imported lazily so the module stays import-clean.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from cue_training.decoders.io import read_jsonl
from cue_training.evaluation.common import ray_fanout
from cue_training.evaluation.common.user_sims import _needs_gpu
from cue_training.evaluation.tau2_bench.runner import _arms_for


def _label(variant: str, arm: str, episode: str) -> str:
    return f"{variant}__{arm}__{episode}".replace(":", "-").replace("/", "_")


def build_units(cases_dir: str, variants: list[str], arms: list[str],
                episodes: list[str] | None) -> list[tuple[str, str, str]]:
    eps = episodes or [r["episode_id"] for r in read_jsonl(str(Path(cases_dir) / "cases.paired.jsonl"))]
    return [(v, a, e) for v in variants for a in _arms_for(v, arms) for e in eps]


def make_cmd_fn(args: argparse.Namespace):
    def make_cmd(unit: tuple[str, str, str], endpoint: str) -> list[str]:
        variant, arm, episode = unit
        cmd = [args.cue_py, "-m", "cue_training.evaluation.tau2_bench.main", "run",
               "--cases_dir", args.cases_dir,
               "--out_dir", str(Path(args.out_dir) / "units" / _label(variant, arm, episode)),
               "--variants", variant, "--arms", arm, "--episode_ids", episode,
               "--agent_llm", args.agent_llm, "--sim_model", args.sim_model]
        ray_fanout.with_flag(cmd, "--tau2_path", args.tau2_path)
        ray_fanout.with_flag(cmd, "--soft_adapter", args.soft_adapter)
        ray_fanout.with_flag(cmd, "--hard_decoder_dir", args.hard_decoder_dir)
        ray_fanout.with_flag(cmd, "--base_local_path", args.base_local_path)
        ray_fanout.with_flag(cmd, "--personas", args.personas)
        # Only route through the sidecar when the variant needs a GPU.
        if _needs_gpu(variant):
            cmd += ["--sidecar_url", endpoint]
        return cmd
    return make_cmd


def run(args: argparse.Namespace) -> None:
    import ray

    ray.init(address=args.ray_address)
    units = build_units(args.cases_dir, args.variants, args.arms, args.episodes)
    model_args = ["--sim_model", args.sim_model]
    ray_fanout.with_flag(model_args, "--soft_adapter", args.soft_adapter)
    ray_fanout.with_flag(model_args, "--soft_model_name", args.soft_model_name)
    ray_fanout.with_flag(model_args, "--semantic_model", args.semantic_model)
    ray_fanout.with_flag(model_args, "--hard_decoder_dir", args.hard_decoder_dir)
    ray_fanout.with_flag(model_args, "--base_local_path", args.base_local_path)
    env_vars = ray_fanout.collect_env_vars()

    needs_gpu = any(_needs_gpu(v) for v in args.variants)
    actors, endpoints = ([], [])
    if needs_gpu:
        actors, endpoints = ray_fanout.start_sidecars(
            ray=ray, cue_py=args.cue_py, cue_root=args.cue_root, model_args=model_args,
            replicas=args.sidecar_replicas, base_port=args.sidecar_base_port,
            warmup_sec=args.sidecar_warmup_sec, env_vars=env_vars,
        )
        print(f"[tau2 ray] sidecars: {endpoints}", flush=True)

    print(f"[tau2 ray] fanning out {len(units)} units (forwarding env: {sorted(env_vars)})", flush=True)
    results = ray_fanout.run_units(
        ray=ray, units=units, make_cmd=make_cmd_fn(args), endpoints=endpoints,
        cpus_per_task=args.cpus_per_task, cwd=args.cue_root, cue_root=args.cue_root,
        env_vars=env_vars,
    )
    n_ok = sum(1 for r in results if r["rc"] == 0)
    print(f"[tau2 ray] {n_ok}/{len(results)} units ok", flush=True)

    out_dir = Path(args.out_dir)
    unit_dirs = [out_dir / "units" / _label(*u) for u in units]
    n = ray_fanout.merge_jsonl([d / "results.jsonl" for d in unit_dirs], out_dir / "results.jsonl")
    ray_fanout.merge_jsonl([d / "rollout.tau2.jsonl" for d in unit_dirs], out_dir / "rollout.tau2.jsonl")
    (out_dir / "run_summary.json").write_text(
        json.dumps({"units": len(units), "ok": n_ok, "merged_results": n}, indent=2), encoding="utf-8")
    for a in actors:
        ray.kill(a)
    print(f"[tau2 ray] merged {n} results -> {out_dir/'results.jsonl'}", flush=True)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Ray fan-out driver for tau2-bench.")
    p.add_argument("--cases_dir", required=True)
    p.add_argument("--out_dir", required=True)
    p.add_argument("--ray_address", default="auto")
    p.add_argument(
        "--cue_root",
        default=os.environ.get("CUE_ROOT", str(Path(__file__).resolve().parents[2])),
    )
    p.add_argument("--cue_py", default=os.environ.get("CUE_PY", "python"))
    p.add_argument("--tau2_path", default=None)
    p.add_argument("--variants", nargs="+",
                   default=["soft_decoder", "baseline:realusersim", "base_api"])
    p.add_argument("--arms", nargs="+", default=["paired", "sample_shuffled", "dataset_mean"])
    p.add_argument("--episodes", nargs="*", default=None, help="Episode-id subset (default: all).")
    p.add_argument("--agent_llm", default="gpt-5.2")
    p.add_argument("--sim_model", default="gpt-5.4-mini")
    p.add_argument("--soft_adapter", default=None)
    p.add_argument("--soft_model_name", default="meta-llama/Llama-3.1-8B-Instruct",
                   help="Base LM the soft-decoder adapter was trained on (sidecar loads this).")
    p.add_argument("--semantic_model", default="intfloat/e5-base-v2")
    p.add_argument("--hard_decoder_dir", default=None)
    p.add_argument("--base_local_path", default=None,
                   help="HF model for base_local (default: sidecar's Llama-3.1-8B-Instruct).")
    p.add_argument("--personas", default=None)
    p.add_argument("--cpus_per_task", type=float, default=1.0)
    p.add_argument("--sidecar_replicas", type=int, default=1)
    p.add_argument("--sidecar_base_port", type=int, default=8713)
    p.add_argument("--sidecar_warmup_sec", type=int, default=45)
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    run(parse_args(argv))


if __name__ == "__main__":
    main()
