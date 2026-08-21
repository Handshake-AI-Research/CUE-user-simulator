"""SimulatorArena writing/math adapter."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path
from typing import Any

from cue_training.evaluation.rollouts.artifacts import promote_final_artifacts
from cue_training.evaluation.rollouts.backends.base import BackendContext, BenchmarkBackend
from cue_training.evaluation.rollouts.config import ConfigError
from cue_training.runlog.log import log


class SimArenaBackend(BenchmarkBackend):
    name = "simulatorarena"

    def prepare(self, ctx: BackendContext) -> dict[str, Any]:
        domain = ctx.job.domain
        domains = ctx.cfg.benchmarks.simulatorarena
        if domain not in domains or not domains[domain].enabled:
            raise ConfigError(f"simulatorarena domain {domain!r} disabled/missing in config")
        dcfg = domains[domain]
        if dcfg.annotations is None:
            raise ConfigError(f"simulatorarena.{domain}.annotations unset")
        if not Path(dcfg.annotations).is_file() and not ctx.dry_run:
            raise ConfigError(f"simulatorarena.{domain}.annotations missing: {dcfg.annotations}")
        embeddings = dcfg.embeddings
        if ctx.job.needs_cue_embeddings and (
            embeddings is None or not Path(embeddings).is_file()
        ):
            if not ctx.dry_run:
                raise ConfigError(
                    f"simulatorarena.{domain}.embeddings required for {ctx.job.method}: {embeddings}"
                )
        if not ctx.job.needs_cue_embeddings:
            # joint training and non-CUE methods do not consume exported vectors.
            embeddings = ctx.scratch_dir / "dummy_embeddings.json"
            if not ctx.dry_run:
                ctx.scratch_dir.mkdir(parents=True, exist_ok=True)
                embeddings.write_text("[]\n", encoding="utf-8")
        from cue_training.evaluation.simulatorarena.data import resolve_profiles_dir

        profiles = resolve_profiles_dir(dcfg.profiles_dir, domain)
        if profiles is None and dcfg.profiles_dir is not None:
            raise ConfigError(
                f"simulatorarena.{domain}.profiles_dir missing or not a directory: {dcfg.profiles_dir}"
            )
        if profiles is not None:
            log("rollouts", f"simulatorarena.{domain}: profiles_dir={profiles}")
        else:
            log(
                "rollouts",
                f"simulatorarena.{domain}: no profiles_dir (persona arms need shipped "
                "user_simulator_profiles; init the SimulatorArena submodule)",
            )
        return {
            "annotations": str(dcfg.annotations),
            "embeddings": str(embeddings),
            "background": str(dcfg.background) if dcfg.background else None,
            "profiles_dir": str(profiles) if profiles else None,
            "domain": domain,
        }

    def run(self, ctx: BackendContext, prep: dict[str, Any]) -> dict[str, Any]:
        scratch = ctx.scratch_dir
        scratch.mkdir(parents=True, exist_ok=True)
        replace_arms = list(ctx.replace_arms or [])
        out_dir = (scratch / ".arm_rerun") if replace_arms else scratch
        out_dir.mkdir(parents=True, exist_ok=True)
        shards_dir = out_dir / "shards"
        shards_dir.mkdir(parents=True, exist_ok=True)
        sim = ctx.cfg.simulator(ctx.job.simulator)
        cmd = [
            sys.executable, "-m", "cue_training.evaluation.simulatorarena.main", "run",
            "--annotations", prep["annotations"],
            "--embeddings", prep["embeddings"],
            "--out_dir", str(out_dir),
            "--domain", prep["domain"],
            "--variants", ctx.job.variant,
            "--arms", *ctx.job.arms,
            "--assistant_model",
            ("dataset" if ctx.cfg.assistant.match_dataset else ctx.cfg.assistant.model),
            "--base_api_model", sim.model,
            "--sim_model", sim.model if sim.kind == "api" else f"hosted_vllm/{sim.served_name or sim.model}",
            "--sim_api_key_env", sim.api_key_env,
            "--sim_max_tokens", str(sim.sampling.max_tokens),
            "--sim_temperature", str(sim.sampling.temperature),
            "--sim_top_p", str(sim.sampling.top_p),
            "--baseline_output_dir", str(ctx.cfg.baselines.artifact_root),
            "--device", "cuda",
            "--concurrency", str(ctx.cfg.resources.concurrency),
            "--error_log", str(
                ctx.paths.run_dir
                / "logs"
                / f"simarena_{prep['domain']}_{ctx.job.method}_{ctx.job.simulator}_episode_errors.jsonl"
            ),
        ]
        if prep.get("background"):
            cmd += ["--background", prep["background"]]
        if prep.get("profiles_dir"):
            cmd += ["--profiles_dir", prep["profiles_dir"]]
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
            cmd += ["--max_conversations", str(ctx.limit)]
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
        # personas.jsonl is a small input sidecar written to the main job dir.
        personas = ctx.job_dir / "personas.jsonl"
        if personas.is_file():
            cmd += ["--personas", str(personas)]

        if ctx.dry_run:
            log("rollouts.simarena", f"dry-run run: {' '.join(cmd)}")
            return {"prep": prep, "shards": [], "cmd": cmd}

        log("rollouts.simarena", f"run: {' '.join(cmd)}")
        subprocess.run(cmd, check=True, cwd=str(ctx.cfg.root))  # noqa: S603
        results = out_dir / "results.jsonl"
        shard_name = "shard_000.jsonl"
        if results.is_file():
            (shards_dir / shard_name).write_bytes(results.read_bytes())
        # Keep only final artifacts in the main outputs dir; native/cases/shards stay in scratch.
        promote_final_artifacts(
            out_dir,
            ctx.job_dir,
            ["results.jsonl", "rollout.simulatorarena.jsonl"],
            replace_arms=replace_arms or None,
        )
        return {"prep": prep, "shards": [shard_name]}
