"""Phase DAG: validate → train → prepare → serve → rollout → merge → metrics."""

from __future__ import annotations

import json
import signal
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from cue_training.baselines.common.manifest import IncompleteArtifactError, train_paper, validate_artifacts
from cue_training.evaluation.rollouts.artifacts import (
    JobSpec,
    atomic_write_json,
    ensure_run_paths,
    job_complete,
    write_run_manifest,
)
from cue_training.evaluation.rollouts.backends.base import BackendContext, get_backend
from cue_training.evaluation.rollouts.config import ConfigError, RolloutsConfig, write_resolved_config
from cue_training.evaluation.rollouts.planning import (
    _probe_port,
    jobs_for_methods,
    pack_job_waves,
    plan_serving,
    policy_served_name,
    sidecar_key,
    simulator_vllm,
)
from cue_training.evaluation.rollouts.ppol_artifacts import (
    PPOL_TRAIN_DOMAIN,
    assign_personas_from_pool,
    load_persona_pool,
    persona_pool_path,
    ppol_artifacts_dir,
    ppol_best_program_path,
    ppol_gen_model,
    ppol_n_personas,
    ppol_sim_model,
    sidecar_covers_as_is,
    tau2_personas_path,
    update_persona_pool_from_sidecar,
)
from cue_training.evaluation.rollouts.recipes import get_recipe
from cue_training.evaluation.rollouts.sidecar import SidecarSupervisor, sidecar_args_for_job
from cue_training.evaluation.rollouts.vllm import VllmSupervisor
from cue_training.evaluation.tau2_bench.personas import PERSONA_BASELINES as TAU2_PERSONA_BASELINES
from cue_training.evaluation.tau2_bench.personas import sidecar_covers_arms
from cue_training.runlog.log import log, warn


class Orchestrator:
    def __init__(
        self,
        cfg: RolloutsConfig,
        *,
        rollout_type: str,
        methods: list[str],
        train: bool = False,
        force_train: bool = False,
        dry_run: bool = False,
        resume: bool = False,
        limit: int | None = None,
        run_id: str | None = None,
        force_prep: bool = False,
        fresh_sidecar: bool = False,
        arms: list[str] | None = None,
        seed: int = 0,
        allow_hash_mismatch: bool = False,
    ) -> None:
        self.cfg = cfg
        self.rollout_type = rollout_type
        self.methods = methods
        self.train = train
        self.force_train = force_train
        self.dry_run = dry_run
        self.resume = resume
        self.limit = limit
        self.run_id = run_id
        self.force_prep = force_prep
        self.fresh_sidecar = fresh_sidecar
        self.arms = list(arms) if arms else None
        self.seed = int(seed)
        self.allow_hash_mismatch = bool(allow_hash_mismatch)
        self.jobs = jobs_for_methods(
            cfg,
            rollout_type=rollout_type,
            methods=methods,
            train=train,
            arms=self.arms,
        )
        # Probe live servers when actually running (route around orphans / reuse compatible);
        # skip probing for dry-run planning.
        self.serve_plan = plan_serving(cfg, self.jobs, probe=(None if dry_run else _probe_port))
        self.paths = ensure_run_paths(
            cfg.resources.output_root,
            config_hash=cfg.config_hash(),
            run_id=run_id,
            resume=resume,
            work_root=cfg.resources.storage_root,
            # Arm-subset reruns already allow a mismatch; --allow-hash-mismatch covers
            # safe knobs like decode_batch_size when resuming a fixed --run-id.
            allow_hash_mismatch=bool(self.arms and resume) or self.allow_hash_mismatch,
        )
        self.vllm = VllmSupervisor(cfg, log_dir=self.paths.run_dir / "logs", dry_run=dry_run)
        self.sidecar = SidecarSupervisor(log_dir=self.paths.run_dir / "logs", dry_run=dry_run)
        self._install_signal_handlers()

    def _install_signal_handlers(self) -> None:
        def _handler(signum: int, _frame: Any) -> None:
            warn("rollouts", f"received signal {signum}; stopping owned servers")
            self.cleanup()
            raise SystemExit(128 + signum)

        for sig in (signal.SIGINT, signal.SIGTERM):
            try:
                signal.signal(sig, _handler)
            except Exception:  # noqa: BLE001
                pass

    def cleanup(self) -> None:
        self.sidecar.stop_owned()
        self.vllm.stop_owned()

    def describe(self) -> dict[str, Any]:
        return {
            "run_id": self.paths.run_id,
            "config_hash": self.paths.config_hash,
            "rollout_type": self.rollout_type,
            "methods": self.methods,
            "arms": self.arms,
            "train": self.train,
            "force_train": self.force_train,
            "dry_run": self.dry_run,
            "resume": self.resume,
            "limit": self.limit,
            "seed": self.seed,
            "benchmarks": self.cfg.enabled_benchmarks(),
            "jobs": [asdict(j) for j in self.jobs],
            "serve_groups": [
                [
                    {
                        "name": r.name,
                        "model": r.model,
                        "served_name": r.served_name,
                        "port": r.port,
                        "gpus": list(r.gpus),
                    }
                    for r in group
                ]
                for group in self.serve_plan.groups
            ],
            "sidecar_ports": self.serve_plan.sidecar_ports,
            "output_root": str(self.paths.run_dir),
        }

    def validate(self) -> None:
        if not self.cfg.enabled_benchmarks():
            raise ConfigError("no benchmarks enabled in config")
        if self.rollout_type == "cue":
            if "general" in self.methods and self.cfg.cue.joint_training_checkpoint is None and self.cfg.cue.decoder_dir is None:
                msg = "cue general requires cue.joint_training_checkpoint or cue.decoder_dir"
                if self.dry_run:
                    print(f"[rollouts] WARN: {msg}", file=sys.stderr)
                else:
                    raise ConfigError(msg)
            if "refined" in self.methods:
                if not self.cfg.cue.refinement_checkpoints and self.cfg.cue.decoder_dir is None:
                    raise ConfigError("cue refined requires at least one refinement checkpoint")
            if "proposals" in self.methods and self.cfg.cue.data_annotation_config is None:
                warn("rollouts", "cue proposals: data_annotation_config unset; assuming precomputed sidecars")

    def train_selected(self) -> None:
        if self.rollout_type != "baseline" or not self.train:
            return
        # Non-ppol: once per method. PPOL: one train_ppol.sh fan-out over all selected sims.
        trained: set[str] = set()
        ppol_sim_models: list[str] = []
        ppol_vllm_base: str | None = None
        for job in self.jobs:
            if not job.train:
                continue
            recipe = get_recipe("baseline", job.method)
            if not recipe.train_supported:
                continue
            if job.method == "ppol":
                sim = self.cfg.simulator(job.simulator)
                sim_model = ppol_sim_model(sim)
                if sim_model not in ppol_sim_models:
                    ppol_sim_models.append(sim_model)
                req = simulator_vllm(sim)
                if req is not None and not self.dry_run and ppol_vllm_base is None:
                    ppol_vllm_base = self.vllm.ensure(req)
                continue
            if job.method in trained:
                continue
            overrides = dict(self.cfg.baselines.paper_overrides.get(job.method) or {})
            artifacts = train_paper(
                job.method,
                artifact_root=self.cfg.baselines.artifact_root,
                data_path=self.cfg.baselines.train_data_path,
                jsonl_names=self.cfg.baselines.train_jsonl_names,
                force=self.force_train,
                dry_run=self.dry_run,
                paper_overrides=overrides,
                repo_root=self.cfg.root,
            )
            if not self.dry_run and job.method != "base":
                validate_artifacts(job.method, artifacts, require_complete=True)
            trained.add(job.method)

        if not ppol_sim_models:
            return
        overrides = dict(self.cfg.baselines.paper_overrides.get("ppol") or {})
        overrides["domain"] = PPOL_TRAIN_DOMAIN
        overrides["sim_models"] = ppol_sim_models
        overrides.setdefault(
            "gen_model",
            overrides.get("mutation_model") or ppol_gen_model(overrides),
        )
        overrides.setdefault("mutation_model", overrides["gen_model"])
        overrides.setdefault("agent_model", self.cfg.assistant.model)
        if ppol_vllm_base is not None:
            overrides["hosted_vllm_api_base"] = ppol_vllm_base
        train_paper(
            "ppol",
            artifact_root=self.cfg.baselines.artifact_root,
            data_path=self.cfg.baselines.train_data_path,
            jsonl_names=self.cfg.baselines.train_jsonl_names,
            force=self.force_train,
            dry_run=self.dry_run,
            paper_overrides=overrides,
            repo_root=self.cfg.root,
        )
        if not self.dry_run:
            for sim_model in ppol_sim_models:
                validate_artifacts(
                    "ppol",
                    ppol_artifacts_dir(
                        self.cfg.baselines.artifact_root,
                        sim_model,
                        domain=PPOL_TRAIN_DOMAIN,
                    ),
                    require_complete=True,
                )

    def _ppol_artifacts_for_job(self, job: JobSpec) -> Path:
        sim = self.cfg.simulator(job.simulator)
        return ppol_artifacts_dir(
            self.cfg.baselines.artifact_root,
            ppol_sim_model(sim),
            domain=PPOL_TRAIN_DOMAIN,
        )

    def _ppol_best_program_for_job(self, job: JobSpec) -> Path:
        sim = self.cfg.simulator(job.simulator)
        return ppol_best_program_path(
            self.cfg.baselines.artifact_root,
            ppol_sim_model(sim),
            domain=PPOL_TRAIN_DOMAIN,
            override=self.cfg.baselines.ppol_best_program,
        )

    def _seeded_persona_pool(self, artifacts: Path) -> Path:
        pool = persona_pool_path(artifacts)
        return pool.with_name(f"{pool.stem}.seed-{self.seed}{pool.suffix}")

    def _ensure_baseline_artifacts(self, job: JobSpec) -> None:
        if job.rollout_type != "baseline" or job.method == "base":
            return
        recipe = get_recipe("baseline", job.method)
        if not recipe.required_artifacts and not recipe.train_supported:
            return
        artifacts = self.cfg.baselines.artifact_root / job.method
        if job.method == "ppol":
            if self.cfg.baselines.ppol_best_program:
                if not self.cfg.baselines.ppol_best_program.is_file():
                    raise IncompleteArtifactError(
                        f"ppol best_program missing: {self.cfg.baselines.ppol_best_program}"
                    )
                return
            # Always the tau2-trained per-sim freeze (ppol/retail_airline_<simtag>).
            artifacts = self._ppol_artifacts_for_job(job)
        try:
            validate_artifacts(job.method, artifacts, require_complete=True)
        except IncompleteArtifactError:
            if self.train:
                raise
            raise IncompleteArtifactError(
                f"{job.method}: artifacts incomplete at {artifacts}. Re-run with --train."
            ) from None

    def _cue_prepare(self, job: JobSpec) -> None:
        """Resolve decoder checkpoint / proposal sidecars for CUE modes."""

        if job.rollout_type != "cue":
            return
        from cue_training.evaluation.common.population_embeddings import ensure_population_embeddings

        ensure_population_embeddings(
            self.cfg, job, dry_run=self.dry_run, force=self.force_prep
        )
        if job.method == "general":
            ckpt = self.cfg.cue.joint_training_checkpoint or self.cfg.cue.decoder_dir
            if ckpt is None:
                raise ConfigError("cue general missing joint_training_checkpoint/decoder_dir")
        if job.method == "refined":
            ckpt = self.cfg.cue.refinement_checkpoints.get(job.simulator) or self.cfg.cue.decoder_dir
            if ckpt is None:
                raise ConfigError(f"cue refined missing refinement checkpoint for {job.simulator}")
        if job.method == "proposals":
            self._prepare_proposals(job)

    def _prepare_proposals(self, job: JobSpec) -> None:
        """data annotation extraction on real eval conversations → RealUserSim persona sidecars."""

        personas = self.paths.job_dir(job) / "personas.jsonl"
        if personas.is_file() and not self.force_prep:
            log("rollouts", f"proposals: reusing sidecar {personas}")
            return
        if self.dry_run:
            log("rollouts", f"dry-run proposals: would extract sidecars -> {personas}")
            return
        if self.cfg.cue.data_annotation_config is None:
            warn("rollouts", "cue proposals: data_annotation_config unset; expecting precomputed personas.jsonl")
            return
        from cue_training.preprocessing.eval_profiles import build_eval_profiles

        sim = self.cfg.simulator(job.simulator)
        if job.benchmark == "tau2":
            cases = self.cfg.benchmarks.tau2.cases_dir
            src = None
            if cases and (Path(cases) / "cases.paired.jsonl").is_file():
                src = Path(cases) / "cases.paired.jsonl"
            else:
                src = self.cfg.benchmarks.tau2.normalized
            if src is None:
                raise ConfigError("proposals/tau2 needs normalized or prepared cases")
            build_eval_profiles(
                config=self.cfg.cue.data_annotation_config,
                benchmark="tau2",
                input_path=src,
                output=personas,
                rollout_sim=sim.model,
                max_cases=int(self.limit or 0),
            )
        elif job.benchmark == "simulatorarena":
            dcfg = self.cfg.benchmarks.simulatorarena[job.domain]
            if dcfg.annotations is None:
                raise ConfigError(f"proposals/{job.domain} missing annotations")
            build_eval_profiles(
                config=self.cfg.cue.data_annotation_config,
                benchmark="simarena",
                input_path=dcfg.annotations,
                output=personas,
                rollout_sim=sim.model,
                domain=job.domain,
                max_cases=int(self.limit or 0),
            )
        elif job.benchmark == "prism":
            src = self.cfg.benchmarks.prism.normalized
            if src is None:
                raise ConfigError("proposals/prism needs prism.normalized")
            log("rollouts", f"proposals/prism: data annotation extraction -> {personas} from {src}")
            build_eval_profiles(
                config=self.cfg.cue.data_annotation_config,
                benchmark="prism",
                input_path=src,
                output=personas,
                rollout_sim=sim.model,
                max_cases=int(self.limit or 0),
            )
        else:
            raise ConfigError(f"proposals: unsupported benchmark {job.benchmark!r}")

    def _persona_extract_llm(self, method: str | None = None) -> tuple[str, str, str | None]:
        """LLM used to distill eval personas (RealUserSim: GPT-4o; else API sim / assistant)."""

        if method in ("realusersim", "realusersim_paired_noex"):
            return "gpt-4o", "OPENAI_API_KEY", None
        api = next((s for s in self.cfg.simulators if s.kind == "api"), None)
        if api is not None:
            return api.model, api.api_key_env, api.api_base
        return self.cfg.assistant.model, self.cfg.assistant.api_key_env, None

    def _prepare_tau2_personas(self, job: JobSpec, prep: dict[str, Any]) -> None:
        """Build tau2 personas.jsonl so paired / shuffled / diverse arms actually differ.

        Tau2 cases have no per-episode profile; without this sidecar USP (and other persona
        baselines) fall back to independent draws from the train profile pool.
        """

        if job.benchmark != "tau2" or job.method not in TAU2_PERSONA_BASELINES:
            return
        out = self.paths.job_dir(job) / "personas.jsonl"
        from cue_training.evaluation.rollouts.recipes import get_recipe

        recipe = get_recipe(job.rollout_type, job.method)
        harness_variant = recipe.variant
        if not self.force_prep and sidecar_covers_arms(out, harness_variant, job.arms):
            log("rollouts", f"{job.method}: reusing personas sidecar {out}")
            return
        cases_dir = Path(prep.get("cases_dir") or "")
        cases = cases_dir / "cases.paired.jsonl"
        if not cases.is_file():
            # Non-CUE prep may only write as_is; fall back to any cases.*.jsonl with conversations.
            candidates = sorted(cases_dir.glob("cases.*.jsonl")) if cases_dir.is_dir() else []
            cases = next((p for p in candidates if p.stat().st_size > 0), cases)
        if self.dry_run:
            log("rollouts", f"dry-run {job.method}: would extract personas -> {out} from {cases}")
            return
        if not cases.is_file():
            warn("rollouts", f"{job.method}: no tau2 cases for persona extraction ({cases})")
            return
        from argparse import Namespace

        from cue_training.evaluation.tau2_bench.personas import run as run_personas

        model, key_env, api_base = self._persona_extract_llm(job.method)
        log("rollouts", f"{job.method}: extracting tau2 personas -> {out} (model={model})")
        run_personas(
            Namespace(
                cases=str(cases),
                out=str(out),
                variants=[job.method],
                sim_model=model,
                api_key_env=key_env,
                api_base=api_base,
                batch_size=50,
                limit=self.limit,
                seed=self.seed,
            )
        )

    def _ppol_tau2_cases_file(self) -> Path | None:
        """Locate (or prep) tau2 cases used to build the shared PPOL persona pool."""

        tcfg = self.cfg.benchmarks.tau2
        if tcfg.normalized is None or tcfg.tau2_path is None:
            return None
        cases_dir = Path(
            tcfg.cases_dir
            or (self.cfg.resources.storage_root or self.cfg.resources.output_root)
            / "ppol_tau2_cases"
        )
        cases_dir.mkdir(parents=True, exist_ok=True)
        existing = sorted(
            p for p in cases_dir.glob("cases.*.jsonl") if p.is_file() and p.stat().st_size > 0
        )
        if existing:
            paired = cases_dir / "cases.paired.jsonl"
            return paired if paired.is_file() else existing[0]
        if self.dry_run:
            return cases_dir / "cases.paired.jsonl"
        import subprocess
        import sys

        embeddings = cases_dir / "dummy_embeddings.json"
        embeddings.write_text("[]\n", encoding="utf-8")
        cmd = [
            sys.executable, "-m", "cue_training.evaluation.tau2_bench.main", "prep",
            "--normalized", str(tcfg.normalized),
            "--embeddings", str(embeddings),
            "--out_dir", str(cases_dir),
            "--tau2_path", str(tcfg.tau2_path),
            "--arms", "as_is",
            "--require_embeddings", "0",
        ]
        log("rollouts", f"ppol: preparing tau2 cases for persona pool -> {cases_dir}")
        subprocess.run(cmd, check=True, cwd=str(self.cfg.root))  # noqa: S603
        paired = cases_dir / "cases.paired.jsonl"
        if paired.is_file():
            return paired
        candidates = sorted(
            p for p in cases_dir.glob("cases.*.jsonl") if p.is_file() and p.stat().st_size > 0
        )
        return candidates[0] if candidates else None

    def _ensure_ppol_persona_pool(
        self,
        job: JobSpec,
        *,
        sim_api_base: str | None = None,
    ) -> Path:
        """Generate tau2 G(c,D,N) personas once per sim into the artifact dir pool."""

        artifacts = (
            self.cfg.baselines.ppol_best_program.parent
            if self.cfg.baselines.ppol_best_program
            else self._ppol_artifacts_for_job(job)
        )
        artifacts.mkdir(parents=True, exist_ok=True)
        pool = self._seeded_persona_pool(artifacts)
        if not self.force_prep and pool.is_file() and pool.stat().st_size > 0:
            return pool
        best = self._ppol_best_program_for_job(job)
        if self.dry_run:
            log("rollouts", f"dry-run ppol: would build persona pool -> {pool} from {best}")
            return pool
        if not best.is_file():
            raise IncompleteArtifactError(f"ppol best_program missing for persona pool: {best}")
        cases = self._ppol_tau2_cases_file()
        if cases is None or not cases.is_file():
            raise ConfigError(
                "ppol: cannot build persona pool without tau2 cases "
                "(set benchmarks.tau2.normalized + tau2_path)"
            )
        tau2_out = tau2_personas_path(artifacts)
        sim = self.cfg.simulator(job.simulator)
        from argparse import Namespace

        from cue_training.evaluation.tau2_bench.ppol_personas import run as run_ppol_personas

        ppol_ov = dict(self.cfg.baselines.paper_overrides.get("ppol") or {})
        gen = ppol_gen_model(ppol_ov)
        n = ppol_n_personas(ppol_ov)
        tau2_path = self.cfg.benchmarks.tau2.tau2_path
        log(
            "rollouts",
            f"ppol: generating tau2 personas -> {tau2_out} (gen={gen}, N={n})",
        )
        run_ppol_personas(
            Namespace(
                cases=str(cases),
                out=str(tau2_out),
                best_program=str(best),
                n=n,
                gen_model=gen,
                sim_model=None,
                tau2_path=str(tau2_path) if tau2_path else None,
                api_base=sim.api_base or sim_api_base,
                api_key_env=sim.api_key_env,
                workers=8,
                limit=self.limit,
                seed=self.seed,
            )
        )
        update_persona_pool_from_sidecar(pool, tau2_out)
        return pool

    def _ppol_episode_ids(self, job: JobSpec, prep: dict[str, Any]) -> list[str]:
        """Episode ids for the current job (for assigning the shared persona pool)."""

        if job.benchmark == "tau2":
            cases_dir = Path(prep.get("cases_dir") or "")
            cases = cases_dir / "cases.paired.jsonl"
            if not cases.is_file():
                candidates = sorted(cases_dir.glob("cases.*.jsonl")) if cases_dir.is_dir() else []
                cases = next((p for p in candidates if p.stat().st_size > 0), cases)
            if not cases.is_file():
                return []
            from cue_training.evaluation.common.io import read_jsonl

            ids: list[str] = []
            seen: set[str] = set()
            for row in read_jsonl(str(cases)):
                eid = str(row.get("episode_id") or "")
                if eid and eid not in seen:
                    seen.add(eid)
                    ids.append(eid)
            return ids[: self.limit] if self.limit else ids

        if job.benchmark == "simulatorarena":
            from cue_training.evaluation.simulatorarena.data import (
                _real_conversation,
                annotation_key,
                load_annotations,
            )

            anns = load_annotations(prep["annotations"])
            domain = str(prep.get("domain") or job.domain)
            ids = [
                annotation_key(ann, domain)
                for ann in anns
                if _real_conversation(ann)
            ]
            return ids[: self.limit] if self.limit else ids

        if job.benchmark == "prism":
            from cue_training.evaluation.common.io import read_jsonl

            path = prep.get("normalized") or self.cfg.benchmarks.prism.normalized
            if not path or not Path(path).is_file():
                return []
            ids = []
            seen: set[str] = set()
            for row in read_jsonl(str(path)):
                eid = str(row.get("episode_id") or row.get("task_id") or "")
                if eid and eid not in seen:
                    seen.add(eid)
                    ids.append(eid)
            return ids[: self.limit] if self.limit else ids
        return []

    def _prepare_ppol_personas(
        self,
        job: JobSpec,
        prep: dict[str, Any],
        *,
        sim_api_base: str | None = None,
    ) -> None:
        """Build job-local personas.jsonl: tau2 G(c,D,N) once per sim, reuse pool elsewhere."""

        if job.method != "ppol":
            return
        out = self.paths.job_dir(job) / "personas.jsonl"
        variant = f"baseline:{job.method}"
        if not self.force_prep and sidecar_covers_as_is(out, variant):
            log("rollouts", f"ppol: reusing personas sidecar {out}")
            return
        if self.dry_run:
            log("rollouts", f"dry-run ppol: would write personas -> {out}")
            return

        best = self._ppol_best_program_for_job(job)
        sim = self.cfg.simulator(job.simulator)
        api_base = sim.api_base or sim_api_base

        if job.benchmark == "tau2":
            cases_dir = Path(prep.get("cases_dir") or "")
            cases = cases_dir / "cases.paired.jsonl"
            if not cases.is_file():
                candidates = sorted(cases_dir.glob("cases.*.jsonl")) if cases_dir.is_dir() else []
                cases = next((p for p in candidates if p.stat().st_size > 0), cases)
            if not cases.is_file():
                warn("rollouts", f"ppol: no tau2 cases for persona generation ({cases})")
                return
            from argparse import Namespace

            from cue_training.evaluation.tau2_bench.ppol_personas import run as run_ppol_personas

            log("rollouts", f"ppol: generating tau2 personas -> {out}")
            ppol_ov = dict(self.cfg.baselines.paper_overrides.get("ppol") or {})
            gen = ppol_gen_model(ppol_ov)
            n = ppol_n_personas(ppol_ov)
            tau2_path = self.cfg.benchmarks.tau2.tau2_path
            run_ppol_personas(
                Namespace(
                    cases=str(cases),
                    out=str(out),
                    best_program=str(best),
                    n=n,
                    gen_model=gen,
                    sim_model=None,
                    tau2_path=str(tau2_path) if tau2_path else None,
                    api_base=api_base,
                    api_key_env=sim.api_key_env,
                    workers=8,
                    limit=self.limit,
                    seed=self.seed,
                )
            )
            artifacts = (
                self.cfg.baselines.ppol_best_program.parent
                if self.cfg.baselines.ppol_best_program
                else self._ppol_artifacts_for_job(job)
            )
            artifacts.mkdir(parents=True, exist_ok=True)
            update_persona_pool_from_sidecar(self._seeded_persona_pool(artifacts), out)
            return

        # Other domains: reuse the tau2-generated persona pool for this simulator.
        pool_path = self._ensure_ppol_persona_pool(job, sim_api_base=api_base)
        pool = load_persona_pool(pool_path)
        if not pool:
            warn("rollouts", f"ppol: empty persona pool at {pool_path}; skipping sidecar")
            return
        episode_ids = self._ppol_episode_ids(job, prep)
        if not episode_ids:
            warn("rollouts", f"ppol: no episode ids for {job.benchmark}/{job.domain}")
            return
        from cue_training.evaluation.common.io import write_jsonl

        rows = assign_personas_from_pool(episode_ids, pool, variant=variant, seed=self.seed)
        write_jsonl(str(out), rows)
        log(
            "rollouts",
            f"ppol: assigned {len(rows)} personas from tau2 pool "
            f"({len(pool)} unique) -> {out}",
        )

    def _serve_for_job(self, job: JobSpec) -> tuple[dict[str, str], str | None]:
        urls: dict[str, str] = {}
        sim = self.cfg.simulator(job.simulator)
        req = simulator_vllm(sim)
        if req is not None and (job.needs_sim_vllm or get_recipe(job.rollout_type, job.method).needs_sim_vllm):
            urls[f"sim:{sim.name}"] = self.vllm.ensure(req)
            urls["sim"] = urls[f"sim:{sim.name}"]
        pref = self.serve_plan.policy_requests.get(job.method)
        if pref is not None and job.needs_policy_vllm:
            urls[job.method] = self.vllm.ensure(pref)
        sidecar_url = None
        if job.needs_sidecar:
            port = self.serve_plan.sidecar_ports.get(sidecar_key(self.cfg, job))
            if port is None:
                raise ConfigError(f"no sidecar port planned for {job.method}/{job.simulator}")
            decoder_dir = None
            if job.rollout_type == "cue":
                resolved = self.cfg.cue.decoder_dir_for(job.method, job.simulator)
                decoder_dir = str(resolved) if resolved else ""
            baseline_artifacts = self.cfg.baselines.artifact_root / job.method
            if job.method == "ppol" and not self.cfg.baselines.ppol_best_program:
                baseline_artifacts = self._ppol_artifacts_for_job(job)
            elif job.method == "ppol" and self.cfg.baselines.ppol_best_program:
                baseline_artifacts = self.cfg.baselines.ppol_best_program.parent
            # For shared CUE sidecars, do not bake the llama vLLM URL into launch args: API sims
            # (gpt/gemini) clear sim_api_base per-request, and a sticky vLLM base 404s those models.
            args = sidecar_args_for_job(
                variant=job.variant,
                sim_model=sim.model if sim.kind == "api" else f"hosted_vllm/{sim.served_name or sim.model}",
                sim_api_base=sim.api_base or (urls.get("sim") if sim.kind == "vllm" else None),
                sim_api_key_env=sim.api_key_env,
                decoder_dir=decoder_dir or None,
                baseline_output_dir=str(self.cfg.baselines.artifact_root),
                baseline_artifacts_dir=str(baseline_artifacts),
                example_retrieval=(
                    {
                        "enabled": self.cfg.cue.example_retrieval_enabled,
                        "k_sessions": self.cfg.cue.example_retrieval_k_sessions,
                        "n_general": self.cfg.cue.example_retrieval_n_general,
                        "n_specific": self.cfg.cue.example_retrieval_n_specific,
                    }
                    if job.rollout_type == "cue"
                    else None
                ),
                vllm_urls=urls,
                usp_vllm_model=policy_served_name(self.cfg, "usp"),
            )
            sidecar_gpu = self._sidecar_gpu()
            sidecar_url = self.sidecar.ensure(port, args, gpu=sidecar_gpu, fresh=self.fresh_sidecar)
        return urls, sidecar_url

    def _sidecar_gpu(self) -> int | None:
        """Which GPU to load the CUE decoder sidecar on: explicit config, else the first GPU not
        used by ANY configured sim vLLM or a planned policy vLLM. Computed run-wide (not per job)
        so a shared sidecar (e.g. cue-general across llama+gpt) avoids whichever sim GPU is busy."""

        if self.cfg.resources.sidecar_gpu is not None:
            return int(self.cfg.resources.sidecar_gpu)
        occupied: set[int] = set()
        for sim in self.cfg.simulators:
            req = simulator_vllm(sim)
            if req is not None:
                occupied |= set(req.gpus)
        for r in self.serve_plan.policy_requests.values():
            occupied |= set(r.gpus)
        free = [g for g in self.cfg.resources.gpus if g not in occupied]
        return int(free[0]) if free else int(self.cfg.resources.gpus[-1])

    def _run_metrics(self, job_dir: Path) -> None:
        if not self.cfg.metrics.enabled:
            return
        rollout = None
        for name in ("rollout.tau2.jsonl", "rollout.simulatorarena.jsonl", "rollout.prism.jsonl"):
            cand = job_dir / name
            if cand.is_file():
                rollout = cand
                break
        if rollout is None:
            return
        import subprocess

        cmd = [
            sys.executable, "-m", "cue_training.evaluation.metrics.run",
            "--baseline", str(rollout),
            "--metrics", ",".join(self.cfg.metrics.enabled),
        ]
        if self.dry_run:
            log("rollouts.metrics", f"dry-run: {' '.join(cmd)}")
            return
        import os

        env = dict(os.environ)
        if self.cfg.metrics.oss_embed_url:
            env["MIRROR_OSS_EMBED_URL"] = self.cfg.metrics.oss_embed_url
        if self.cfg.metrics.oss_gen_url:
            env["MIRROR_OSS_GEN_URL"] = self.cfg.metrics.oss_gen_url
        try:
            subprocess.run(cmd, check=True, cwd=str(self.cfg.root), env=env)  # noqa: S603
            # metrics.run writes under the configured storage root.
            from cue_training.evaluation.metrics.run import _default_run_dir

            summary = _default_run_dir(rollout, None) / "summary.json"
            if summary.is_file():
                (job_dir / "metrics.json").write_bytes(summary.read_bytes())
        except Exception as exc:  # noqa: BLE001
            warn("rollouts.metrics", f"metrics failed: {exc}")

    def _execute_job(self, job: JobSpec) -> dict[str, Any] | None:
        """Prepare + run + finalize one job. Returns summary, or None if skipped."""

        job_dir = self.paths.job_dir(job)
        job_dir.mkdir(parents=True, exist_ok=True)
        write_resolved_config(self.cfg, job_dir / "config.resolved.json")
        # Arm-subset reruns must not be skipped: COMPLETE was written for the full arm set.
        if self.resume and job_complete(job_dir) and not self.arms:
            log("rollouts", f"skip complete job {job_dir}")
            return None
        self._ensure_baseline_artifacts(job)
        self._cue_prepare(job)
        backend = get_backend(job.benchmark)
        work_dir = self.paths.work_dir(job)
        work_dir.mkdir(parents=True, exist_ok=True)
        # Sample / join embeddings BEFORE starting the decoder sidecar. Prep (especially
        # sampled_uncond/sampled_pop) loads the diffusion sampler onto cue.sampler_device;
        # if the sidecar already holds that GPU, sampling and later decode both thrash.
        prep_ctx = BackendContext(
            cfg=self.cfg,
            paths=self.paths,
            job=job,
            job_dir=job_dir,
            dry_run=False,
            resume=self.resume,
            limit=self.limit,
            vllm_urls=None,
            sidecar_url=None,
            work_dir=work_dir,
            force_prep=self.force_prep,
            replace_arms=list(self.arms) if self.arms else None,
        )
        prep = backend.prepare(prep_ctx)
        sidecar_gpu = self._sidecar_gpu()
        sampler_dev = (self.cfg.cue.sampler_device or "").strip()
        if (
            sidecar_gpu is not None
            and sampler_dev
            and sampler_dev.replace("cuda:", "") == str(sidecar_gpu)
        ):
            warn(
                "rollouts",
                f"cue.sampler_device={sampler_dev!r} matches resources.sidecar_gpu={sidecar_gpu}; "
                "move the sampler to a free GPU so diffusion sampling does not contend with decode",
            )
        urls, sidecar_url = self._serve_for_job(job)
        ctx = BackendContext(
            cfg=self.cfg,
            paths=self.paths,
            job=job,
            job_dir=job_dir,
            dry_run=False,
            resume=self.resume,
            limit=self.limit,
            vllm_urls=urls,
            sidecar_url=sidecar_url,
            work_dir=work_dir,
            force_prep=self.force_prep,
            replace_arms=list(self.arms) if self.arms else None,
        )
        self._prepare_tau2_personas(job, prep)
        self._prepare_ppol_personas(job, prep, sim_api_base=urls.get("sim"))
        run_info = backend.run(ctx, prep)
        summary = backend.finalize(ctx, run_info)
        self._run_metrics(job_dir)
        return summary

    def run(self) -> dict[str, Any]:
        description = self.describe()
        if self.dry_run:
            # Pure plan dump first (machine-readable), then soft validation warnings.
            print(json.dumps(description, indent=2, sort_keys=True))
            self.validate()
            return {"run_id": self.paths.run_id, "summaries": [], "plan": description}

        self.validate()
        write_resolved_config(self.cfg, self.paths.run_dir / "config.resolved.json")
        if self.resume and self.arms:
            # In-place arm replace: keep the original run manifest job list; annotate only.
            manifest_path = self.paths.run_dir / "manifest.json"
            if manifest_path.is_file():
                existing = json.loads(manifest_path.read_text(encoding="utf-8"))
                existing["arm_rerun"] = {
                    "arms": list(self.arms),
                    "at": datetime.now(timezone.utc).isoformat(),
                    "current_config_hash": self.cfg.config_hash(),
                }
                atomic_write_json(manifest_path, existing)
            log(
                "rollouts",
                f"in-place arm replace on run_id={self.paths.run_id}: arms={self.arms}",
            )
        else:
            write_run_manifest(self.paths, jobs=self.jobs, extra={"dry_run": False})
        try:
            self.train_selected()
            summaries: list[dict[str, Any]] = []
            waves = pack_job_waves(self.cfg, self.jobs, self.serve_plan.policy_requests)
            for wi, wave in enumerate(waves):
                labels = [f"{j.benchmark}/{j.domain}/{j.method}/{j.simulator}" for j in wave]
                log("rollouts", f"wave {wi + 1}/{len(waves)}: {len(wave)} jobs in parallel: {labels}")
                if len(wave) == 1:
                    summary = self._execute_job(wave[0])
                    if summary is not None:
                        summaries.append(summary)
                else:
                    with ThreadPoolExecutor(max_workers=len(wave)) as pool:
                        futures = {pool.submit(self._execute_job, job): job for job in wave}
                        for fut in as_completed(futures):
                            job = futures[fut]
                            try:
                                summary = fut.result()
                            except Exception as exc:
                                warn(
                                    "rollouts",
                                    f"job failed: {job.benchmark}/{job.domain}/{job.method}/{job.simulator}: {exc}",
                                )
                                raise
                            if summary is not None:
                                summaries.append(summary)
                # Free GPUs held by this wave's policy/sim servers before the next
                # method (userlm ↔ usp share one seed card) tries to claim them.
                if wi + 1 < len(waves):
                    from cue_training.evaluation.rollouts.planning import vllm_requests_for_job

                    needed: list = []
                    for job in waves[wi + 1]:
                        needed.extend(
                            vllm_requests_for_job(
                                self.cfg, job, self.serve_plan.policy_requests
                            )
                        )
                    self.vllm.release_not_needed(needed)
            return {"run_id": self.paths.run_id, "summaries": summaries, "plan": description}
        finally:
            self.cleanup()
