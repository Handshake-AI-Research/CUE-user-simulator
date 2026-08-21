"""Tau2 benchmark adapter: prep once, then invoke tau2-eval run."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path
from typing import Any

from cue_training.evaluation.rollouts.artifacts import promote_final_artifacts
from cue_training.evaluation.rollouts.backends.base import BackendContext, BenchmarkBackend
from cue_training.evaluation.rollouts.config import ConfigError
from cue_training.runlog.log import log


class Tau2Backend(BenchmarkBackend):
    name = "tau2"

    def prepare(self, ctx: BackendContext) -> dict[str, Any]:
        tcfg = ctx.cfg.benchmarks.tau2
        if not tcfg.enabled:
            raise ConfigError("tau2 benchmark disabled in config")
        normalized = tcfg.normalized
        tau2_path = tcfg.tau2_path
        if normalized is None:
            raise ConfigError("tau2.normalized unset")
        if not Path(normalized).is_file() and not ctx.dry_run:
            raise ConfigError(f"tau2.normalized missing: {normalized}")
        if tau2_path is None:
            raise ConfigError("tau2.tau2_path unset")
        if not Path(tau2_path).is_dir() and not ctx.dry_run:
            raise ConfigError(f"tau2.tau2_path missing checkout: {tau2_path}")

        cases_dir = tcfg.cases_dir or (ctx.scratch_dir / "cases")
        cases_dir = Path(cases_dir)
        cases_dir.mkdir(parents=True, exist_ok=True)
        # Reuse any non-empty cases.*.jsonl (base writes cases.as_is.jsonl, CUE writes paired/...),
        # unless --force-prep is set (re-join the embeddings + rebuild cases).
        replace_arms = list(ctx.replace_arms or [])
        if ctx.force_prep:
            if replace_arms:
                for arm in replace_arms:
                    (cases_dir / f"cases.{arm}.jsonl").unlink(missing_ok=True)
            else:
                for stale in cases_dir.glob("cases.*.jsonl"):
                    stale.unlink(missing_ok=True)
        elif not ctx.dry_run:
            if replace_arms:
                if all((cases_dir / f"cases.{arm}.jsonl").is_file() for arm in replace_arms):
                    log("rollouts.tau2", f"reusing prepared cases at {cases_dir}")
                    return {
                        "cases_dir": str(cases_dir),
                        "tau2_path": str(tau2_path),
                        "normalized": str(normalized),
                    }
            elif any(
                path.is_file() and path.stat().st_size > 0
                for path in cases_dir.glob("cases.*.jsonl")
            ):
                log("rollouts.tau2", f"reusing prepared cases at {cases_dir}")
                return {
                    "cases_dir": str(cases_dir),
                    "tau2_path": str(tau2_path),
                    "normalized": str(normalized),
                }

        embeddings = tcfg.embeddings
        if ctx.job.needs_cue_embeddings and (embeddings is None or not Path(embeddings).is_file()):
            raise ConfigError(
                f"tau2 embeddings required for {ctx.job.method} but missing: {embeddings}"
            )
        # joint training and non-CUE methods do not consume exported vectors.
        if not ctx.job.needs_cue_embeddings:
            embeddings = cases_dir / "dummy_embeddings.json"
            if not ctx.dry_run:
                embeddings.write_text("[]\n", encoding="utf-8")

        cmd = [
            sys.executable, "-m", "cue_training.evaluation.tau2_bench.main", "prep",
            "--normalized", str(normalized),
            "--embeddings", str(embeddings),
            "--out_dir", str(cases_dir),
            "--tau2_path", str(tau2_path),
            "--arms", *ctx.job.arms,
            "--require_embeddings", "1" if ctx.job.needs_cue_embeddings else "0",
        ]
        from cue_training.evaluation.common.sampler_arms import sampler_cli_flags

        cmd += sampler_cli_flags(
            checkpoint=ctx.cfg.cue.sampler_checkpoint,
            seed=ctx.cfg.cue.sampler_seed,
            guidance_w=ctx.cfg.cue.sampler_guidance_w,
            ddim_steps=ctx.cfg.cue.sampler_ddim_steps,
            device=ctx.cfg.cue.sampler_device,
            pop_max_k=ctx.cfg.cue.sampler_pop_max_k,
            project_manifold=ctx.cfg.cue.sampler_project_manifold,
        )
        if ctx.dry_run:
            log("rollouts.tau2", f"dry-run prep: {' '.join(cmd)}")
            return {"cases_dir": str(cases_dir), "tau2_path": str(tau2_path), "cmd": cmd}

        log("rollouts.tau2", f"prep: {' '.join(cmd)}")
        subprocess.run(cmd, check=True, cwd=str(ctx.cfg.root))  # noqa: S603
        return {"cases_dir": str(cases_dir), "tau2_path": str(tau2_path), "normalized": str(normalized)}

    def run(self, ctx: BackendContext, prep: dict[str, Any]) -> dict[str, Any]:
        scratch = ctx.scratch_dir
        scratch.mkdir(parents=True, exist_ok=True)
        replace_arms = list(ctx.replace_arms or [])
        # Stage arm-subset reruns so we do not clobber sibling arms in job_dir/scratch.
        out_dir = (scratch / ".arm_rerun") if replace_arms else scratch
        out_dir.mkdir(parents=True, exist_ok=True)
        shards_dir = out_dir / "shards"
        shards_dir.mkdir(parents=True, exist_ok=True)
        sim = ctx.cfg.simulator(ctx.job.simulator)
        shard_name = "shard_000.jsonl"
        shard_path = shards_dir / shard_name

        cmd = [
            sys.executable, "-m", "cue_training.evaluation.tau2_bench.main", "run",
            "--cases_dir", str(prep["cases_dir"]),
            "--out_dir", str(out_dir),
            "--publish_dir", str(ctx.job_dir),
            "--variants", ctx.job.variant,
            "--arms", *ctx.job.arms,
            "--tau2_path", str(prep["tau2_path"]),
            "--agent_llm", ctx.cfg.assistant.model,
            "--base_api_model", sim.model if ctx.job.variant == "base_api" else sim.model,
            "--sim_model", sim.model if sim.kind == "api" else f"hosted_vllm/{sim.served_name or sim.model}",
            "--sim_api_key_env", sim.api_key_env,
            "--sim_max_tokens", str(sim.sampling.max_tokens),
            "--sim_temperature", str(sim.sampling.temperature),
            "--sim_top_p", str(sim.sampling.top_p),
            "--baseline_output_dir", str(ctx.cfg.baselines.artifact_root),
            "--device", "cuda",
            "--concurrency", str(ctx.cfg.resources.concurrency),
        ]
        if replace_arms:
            cmd += ["--replace_arms"]
        if sim.kind == "vllm":
            api_base = sim.api_base or (ctx.vllm_urls or {}).get(f"sim:{sim.name}") or f"http://127.0.0.1:{sim.port}/v1"
            cmd += ["--sim_api_base", api_base, "--vllm_base_url", api_base]
            if ctx.job.variant == "base_local":
                cmd += ["--base_local_path", sim.model]
        elif sim.api_base:
            # Shared external vLLM (kind=api + api_base) still needs the base URL forwarded;
            # without it litellm's hosted_vllm provider 404s on its default endpoint.
            cmd += ["--sim_api_base", sim.api_base]
        if ctx.sidecar_url:
            cmd += ["--sidecar_url", ctx.sidecar_url]
        if ctx.limit is not None:
            cmd += ["--max_episodes", str(ctx.limit)]
        if ctx.job.method in {"usp", "userlm"}:
            urls = ctx.vllm_urls or {}
            if ctx.job.method == "usp" and "usp" in urls:
                from cue_training.evaluation.rollouts.planning import policy_served_name

                cmd += [
                    "--usp_vllm_base_url", urls["usp"],
                    "--usp_vllm_model", policy_served_name(ctx.cfg, "usp"),
                ]
            if ctx.job.method == "userlm" and "userlm" in urls:
                cmd += ["--userlm_vllm_base_url", urls["userlm"], "--userlm_vllm_model", "userlm"]
        if ctx.job.rollout_type == "cue":
            ddir = ctx.cfg.cue.decoder_dir_for(ctx.job.method, ctx.job.simulator)
            if ddir:
                cmd += ["--decoder_dir", str(ddir)]
            cmd += ["--decode_temperature", str(ctx.cfg.cue.decode_temperature),
                    "--decode_top_p", str(ctx.cfg.cue.decode_top_p),
                    "--decode_num_candidates", str(ctx.cfg.cue.decode_num_candidates),
                    "--decode", str(ctx.cfg.cue.decode_mode),
                    "--decode_slot_dedup_jaccard", str(ctx.cfg.cue.decode_slot_dedup_jaccard),
                    "--decode_noop_retries", str(ctx.cfg.cue.decode_noop_retries),
                    "--decode_noop_temperature", str(ctx.cfg.cue.decode_noop_temperature),
                    "--decode_batch_size", str(ctx.cfg.cue.decode_batch_size)]
            from cue_training.evaluation.common.user_sims import example_retrieval_cli_flags

            cmd += example_retrieval_cli_flags(
                enabled=ctx.cfg.cue.example_retrieval_enabled,
                k_sessions=ctx.cfg.cue.example_retrieval_k_sessions,
                n_general=ctx.cfg.cue.example_retrieval_n_general,
                n_specific=ctx.cfg.cue.example_retrieval_n_specific,
            )
            cmd += [
                "--encode_session_preprocess",
                ctx.cfg.cue.session_preprocess_for(ctx.job.domain),
            ]
            if not ctx.job.needs_cue_embeddings:
                cmd += ["--live_encode_trajectories"]
        # personas.jsonl is a small input sidecar written to the main job dir.
        personas = ctx.job_dir / "personas.jsonl"
        if personas.is_file():
            cmd += ["--personas", str(personas)]

        if ctx.dry_run:
            log("rollouts.tau2", f"dry-run run: {' '.join(cmd)}")
            return {"prep": prep, "shards": [], "cmd": cmd}

        log("rollouts.tau2", f"run: {' '.join(cmd)}")
        subprocess.run(cmd, check=True, cwd=str(ctx.cfg.root))  # noqa: S603

        # Capture a shard listing pointing at the merged results for resume isolation.
        results = out_dir / "results.jsonl"
        if results.is_file():
            shard_path.write_bytes(results.read_bytes())
        # Promote / arm-merge final artifacts into the main outputs dir.
        promote_final_artifacts(
            out_dir,
            ctx.job_dir,
            ["results.jsonl", "rollout.tau2.jsonl"],
            replace_arms=replace_arms or None,
        )
        # When scratch == job_dir and we staged under .arm_rerun, also merge into scratch
        # copies if they live at job_dir (promote already wrote job_dir).
        return {"prep": prep, "shards": [shard_name]}
