"""Config validation, recipe expansion, planner, and dry-run coverage."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from cue_training.evaluation.rollouts.artifacts import (
    ArtifactError,
    ensure_run_paths,
    listed_shards,
    merge_jsonl_shards,
    promote_final_artifacts,
    write_job_manifest,
)
from cue_training.evaluation.rollouts.cli import _resolve_config_path, build_parser
from cue_training.evaluation.rollouts.config import ConfigError, load_rollouts_config
from cue_training.evaluation.rollouts.planning import (
    jobs_for_methods,
    pack_job_waves,
    plan_serving,
    simulator_vllm,
)
from cue_training.evaluation.rollouts.recipes import expand_methods, get_recipe
from cue_training.utils.config import load_config


ROOT = Path(__file__).resolve().parents[1]


def _tiny_config(tmp_path: Path, **overrides) -> Path:
    payload = {
        "benchmarks": {
            "tau2": {
                "enabled": True,
                "normalized": str(ROOT / "data/mirrorbench/data/tau_usi/normalized.jsonl"),
                "tau2_path": str(ROOT / "evaluation/tau2_bench/external/tau2-bench"),
            },
            "simulatorarena": {
                "writing": {"enabled": False},
                "math": {"enabled": False},
            },
            "prism": {
                "enabled": True,
                "normalized": str(ROOT / "tests/fixtures/prism/normalized.jsonl"),
            },
        },
        "simulators": [
            {
                "name": "llama",
                "kind": "vllm",
                "model": "meta-llama/Llama-3.1-8B-Instruct",
                "served_name": "llama",
                "gpus": [0],
                "port": 8000,
            },
            {"name": "gpt", "kind": "api", "model": "gpt-4o-mini", "api_key_env": "OPENAI_API_KEY"},
        ],
        "assistant": {"model": "gpt-4o-mini"},
        "resources": {
            "gpus": [0, 1],
            "output_root": str(tmp_path / "out"),
            "start_vllm_script": str(ROOT / "scripts/start_vllm.sh"),
        },
        "cue": {"arms": ["paired", "sample_shuffled"], "decoder_dir": str(tmp_path / "decoder")},
        "baselines": {"artifact_root": str(tmp_path / "baselines")},
        "metrics": {"enabled": []},
    }
    payload.update(overrides)
    path = tmp_path / "rollouts.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def test_load_default_rollouts_config(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("CUE_STORAGE_ROOT", str(tmp_path))
    cfg = load_rollouts_config(ROOT / "configs/rollouts.json", repo_root=ROOT)
    assert cfg.assistant.model
    assert len(cfg.simulators) >= 1
    assert ("tau2", "customer-service") in cfg.enabled_benchmarks()
    assert ("prism", "diversity") in cfg.enabled_benchmarks()
    assert cfg.cue.decode_mode == "greedy"
    assert cfg.cue.decode_temperature == 0.0
    assert cfg.cue.decode_top_p == 0.9
    assert cfg.cue.joint_training_checkpoint == tmp_path / "models/cue/best"
    assert cfg.cue.data_annotation_config == ROOT / "configs/data_annotation_validation.json"
    assert cfg.benchmarks.tau2.tau2_path == (ROOT / "cue_training/evaluation/tau2_bench/external/tau2-bench")
    assert cfg.benchmarks.simulatorarena["writing"].annotations == (
        ROOT / "cue_training/evaluation/simulatorarena/external/SimulatorArena/data/document_creation_annotations.json"
    )


def test_rollout_cli_finds_shipped_config_from_repo_root() -> None:
    assert _resolve_config_path("configs/rollouts.json") == ROOT / "configs/rollouts.json"


def test_standard_config_expands_storage_root(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("CUE_STORAGE_ROOT", str(tmp_path))
    cfg = load_config(ROOT / "configs/joint_training.json")
    assert cfg["data_root"] == str(tmp_path / "data")
    assert cfg["output"] == str(tmp_path / "models/cue")


def test_rollout_cli_accepts_replicate_seed() -> None:
    args = build_parser().parse_args(["cue", "general", "--seed", "3"])
    assert args.seed == 3


def test_session_preprocess_domain_defaults(tmp_path: Path):
    cfg = load_rollouts_config(_tiny_config(tmp_path), repo_root=ROOT)
    assert cfg.cue.session_preprocess_for("writing") == "strip_document"
    assert cfg.cue.session_preprocess_for("math") == "full"
    assert cfg.cue.session_preprocess_for("customer-service") == "full"
    assert cfg.cue.session_preprocess_for("diversity") == "full"


def test_reject_unknown_keys(tmp_path: Path):
    path = _tiny_config(tmp_path)
    raw = json.loads(path.read_text())
    raw["nope"] = 1
    path.write_text(json.dumps(raw), encoding="utf-8")
    with pytest.raises(ConfigError, match="unknown keys"):
        load_rollouts_config(path, repo_root=ROOT)


def test_vllm_simulator_requires_port_and_gpus(tmp_path: Path):
    path = _tiny_config(tmp_path)
    raw = json.loads(path.read_text())
    raw["simulators"] = [{"name": "x", "kind": "vllm", "model": "m"}]
    path.write_text(json.dumps(raw), encoding="utf-8")
    with pytest.raises(ConfigError):
        load_rollouts_config(path, repo_root=ROOT)


def test_expand_methods_all_and_aliases():
    assert expand_methods("baseline", ["all"])[0] == "base"
    assert "usp" in expand_methods("baseline", ["all"])
    assert expand_methods("cue", ["proposals", "general"]) == ["proposals", "general"]
    with pytest.raises(ValueError):
        expand_methods("baseline", ["not-a-method"])


def test_recipe_arms():
    assert get_recipe("baseline", "usp").needs_policy_vllm
    assert not get_recipe("cue", "general").needs_cue_embeddings
    assert get_recipe("cue", "refined").variant == "decoder"


def test_jobs_for_methods_arms_override(tmp_path: Path):
    cfg = load_rollouts_config(_tiny_config(tmp_path), repo_root=ROOT)
    jobs = jobs_for_methods(cfg, rollout_type="cue", methods=["general"], train=False, arms=["sampled_pop"])
    assert jobs
    assert all(j.arms == ["sampled_pop"] for j in jobs)


def test_proposals_arms_never_include_sampled(tmp_path: Path):
    path = _tiny_config(tmp_path)
    raw = json.loads(path.read_text())
    raw["cue"]["arms"] = ["paired", "sample_shuffled", "sampled_uncond", "sampled_pop"]
    path.write_text(json.dumps(raw), encoding="utf-8")
    cfg = load_rollouts_config(path, repo_root=ROOT)
    jobs = jobs_for_methods(cfg, rollout_type="cue", methods=["proposals"], train=False)
    assert jobs
    assert all(j.arms == ["paired", "sample_shuffled"] for j in jobs)
    assert all(not j.needs_cue_embeddings for j in jobs)
    # CLI --arms cannot add sampler arms to proposals either.
    jobs2 = jobs_for_methods(
        cfg,
        rollout_type="cue",
        methods=["proposals"],
        train=False,
        arms=["sampled_pop", "paired"],
    )
    assert all(j.arms == ["paired"] for j in jobs2)


def test_gpu_port_planner(tmp_path: Path):
    cfg = load_rollouts_config(_tiny_config(tmp_path), repo_root=ROOT)
    jobs = jobs_for_methods(cfg, rollout_type="baseline", methods=["base", "userlm"], train=False)
    plan = plan_serving(cfg, jobs)
    assert plan.groups
    assert any(r.name.startswith("sim:") for group in plan.groups for r in group)
    assert any(r.name == "userlm" for group in plan.groups for r in group)


def test_policy_baseline_first_free_gpu_and_no_gpt(tmp_path: Path):
    # tiny config has sims [llama(vllm,gpu0), gpt(api)] and resources.gpus [0,1].
    cfg = load_rollouts_config(_tiny_config(tmp_path), repo_root=ROOT)
    jobs = jobs_for_methods(cfg, rollout_type="baseline", methods=["userlm"], train=False)
    # (b) model-loading baseline collapses to a single (non-gpt) job.
    assert len(jobs) == len([1 for _ in cfg.enabled_benchmarks()])
    assert all(j.simulator != "gpt" for j in jobs)
    # (a) userlm starts no sim vLLM, so with nothing running (probe disabled) it takes gpu0.
    plan = plan_serving(cfg, jobs, probe=None)
    assert plan.policy_requests["userlm"].gpus == (0,)


def test_policy_gpu_avoids_started_sim(tmp_path: Path):
    cfg = load_rollouts_config(_tiny_config(tmp_path), repo_root=ROOT)
    # base starts the llama sim vLLM on gpu0, so userlm's policy must skip to gpu1.
    jobs = jobs_for_methods(cfg, rollout_type="baseline", methods=["base", "userlm"], train=False)
    plan = plan_serving(cfg, jobs, probe=None)
    assert plan.policy_requests["userlm"].gpus == (1,)


def test_policy_reuses_or_avoids_live_server(tmp_path: Path):
    cfg = load_rollouts_config(_tiny_config(tmp_path), repo_root=ROOT)
    jobs = jobs_for_methods(cfg, rollout_type="baseline", methods=["usp"], train=False)

    # An orphan userlm server occupies gpu0:8000 -> usp must skip to gpu1.
    def probe(port: int):
        return ["userlm"] if port == 8000 else None

    plan = plan_serving(cfg, jobs, probe=probe)
    assert plan.policy_requests["usp"].gpus == (1,)

    # If published USP is already live on gpu0:8000, reuse it in place.
    def probe_compatible(port: int):
        return ["wangkevin02/USP"] if port == 8000 else None

    plan2 = plan_serving(cfg, jobs, probe=probe_compatible)
    assert plan2.policy_requests["usp"].gpus == (0,)


def test_policy_falls_back_to_unused_port_when_all_busy(tmp_path: Path):
    cfg = load_rollouts_config(_tiny_config(tmp_path), repo_root=ROOT)
    jobs = jobs_for_methods(cfg, rollout_type="baseline", methods=["usp"], train=False)
    # Both conventional ports are held by incompatible orphans -> must pick an unused port.
    busy = {8000: ["llama"], 8001: ["userlm"]}

    def probe(port: int):
        return busy.get(port)

    plan = plan_serving(cfg, jobs, probe=probe)
    req = plan.policy_requests["usp"]
    assert req.port not in (8000, 8001)
    assert probe(req.port) is None


def test_all_serving_ports_disjoint(tmp_path: Path):
    # 4 GPUs so base(llama sim) + userlm + usp each get their own device/port.
    res = {
        "gpus": [0, 1, 2, 3],
        "output_root": str(tmp_path / "out"),
        "start_vllm_script": str(ROOT / "scripts/start_vllm.sh"),
    }
    cfg = load_rollouts_config(_tiny_config(tmp_path, resources=res), repo_root=ROOT)
    jobs = jobs_for_methods(cfg, rollout_type="baseline", methods=["base", "userlm", "usp"], train=False)
    plan = plan_serving(cfg, jobs, probe=None)
    sim_ports = [r.port for r in map(simulator_vllm, cfg.simulators) if r is not None]
    policy_ports = [r.port for r in plan.policy_requests.values()]
    sidecar_ports = list(plan.sidecar_ports.values())
    all_ports = sim_ports + policy_ports + sidecar_ports
    assert len(all_ports) == len(set(all_ports)), (sim_ports, policy_ports, sidecar_ports)
    assert len(set(policy_ports)) == len(policy_ports)


def test_pack_job_waves_runs_api_with_vllm(tmp_path: Path):
    cfg = load_rollouts_config(_tiny_config(tmp_path), repo_root=ROOT)
    jobs = jobs_for_methods(cfg, rollout_type="baseline", methods=["base"], train=False)
    waves = pack_job_waves(cfg, jobs)
    # Per (benchmark, domain): llama (vLLM) + gpt (API) share a wave.
    tau2_wave = next(w for w in waves if any(j.benchmark == "tau2" for j in w))
    sims = {j.simulator for j in tau2_wave}
    assert sims == {"llama", "gpt"}
    # Distinct domains/benchmarks stay in separate waves (don't fan out everything).
    assert len(waves) >= 2


def test_pack_job_waves_runs_api_baseline_with_policy_gpu(tmp_path: Path):
    """Cross-method packing: API baselines can share a wave with a policy-vLLM method."""

    cfg = load_rollouts_config(_tiny_config(tmp_path), repo_root=ROOT)
    for methods, must_share in (
        (["usp", "realusersim"], {"usp", "realusersim"}),
        (["userlm", "ppol"], {"userlm", "ppol"}),
    ):
        jobs = jobs_for_methods(cfg, rollout_type="baseline", methods=list(methods), train=False)
        plan = plan_serving(cfg, jobs, probe=None)
        waves = pack_job_waves(cfg, jobs, plan.policy_requests)
        tau2_wave = next(w for w in waves if any(j.benchmark == "tau2" for j in w))
        assert must_share <= {j.method for j in tau2_wave}


def test_baseline_jobs_schedule_tau2_before_other_benchmarks(tmp_path: Path):
    cfg = load_rollouts_config(_tiny_config(tmp_path), repo_root=ROOT)
    jobs = jobs_for_methods(cfg, rollout_type="baseline", methods=["base", "userlm"], train=False)
    waves = pack_job_waves(cfg, jobs)
    first_non_tau2 = next(
        (i for i, w in enumerate(waves) if any(j.benchmark != "tau2" for j in w)),
        len(waves),
    )
    assert all(j.benchmark == "tau2" for w in waves[:first_non_tau2] for j in w)
    assert any(j.benchmark == "tau2" for j in jobs)


def test_work_root_splits_scratch_from_outputs(tmp_path: Path):
    from cue_training.evaluation.rollouts.artifacts import JobSpec

    paths = ensure_run_paths(tmp_path / "out", config_hash="h", run_id="r1", work_root=tmp_path / "scratch")
    job = JobSpec(
        rollout_type="baseline",
        method="base",
        benchmark="tau2",
        domain="customer-service",
        simulator="llama",
        arms=["as_is"],
        variant="base_local",
    )
    job_dir = paths.job_dir(job)
    work_dir = paths.work_dir(job)
    assert (tmp_path / "out") in job_dir.parents
    assert (tmp_path / "scratch") in work_dir.parents
    assert job_dir != work_dir
    # promote copies only the named finals; bulky scratch files stay behind.
    work_dir.mkdir(parents=True)
    (work_dir / "results.jsonl").write_text('{"x":1}\n', encoding="utf-8")
    (work_dir / "cases").mkdir()
    promote_final_artifacts(work_dir, job_dir, ["results.jsonl", "rollout.tau2.jsonl"])
    assert (job_dir / "results.jsonl").is_file()
    assert not (job_dir / "cases").exists()


def test_work_root_defaults_to_outputs(tmp_path: Path):
    from cue_training.evaluation.rollouts.artifacts import JobSpec

    paths = ensure_run_paths(tmp_path / "out", config_hash="h", run_id="r1")
    job = JobSpec(
        rollout_type="baseline",
        method="base",
        benchmark="tau2",
        domain="customer-service",
        simulator="llama",
        arms=["as_is"],
        variant="base_local",
    )
    # No work_root -> scratch == main outputs (behavior unchanged).
    assert paths.work_dir(job) == paths.job_dir(job)


def test_resume_hash_mismatch(tmp_path: Path):
    run = ensure_run_paths(tmp_path / "out", config_hash="aaa", run_id="r1", resume=False)
    (run.run_dir / "manifest.json").write_text(json.dumps({"config_hash": "aaa", "jobs": []}), encoding="utf-8")
    with pytest.raises(ArtifactError, match="hash mismatch"):
        ensure_run_paths(tmp_path / "out", config_hash="bbb", run_id="r1", resume=True)


def test_shard_merge_only_lists_manifest(tmp_path: Path):
    job = tmp_path / "job"
    shards = job / "shards"
    shards.mkdir(parents=True)
    (shards / "shard_000.jsonl").write_text('{"id":1}\n', encoding="utf-8")
    (shards / "stale.jsonl").write_text('{"id":99}\n', encoding="utf-8")
    write_job_manifest(job, {"shards": ["shard_000.jsonl"], "status": "running"})
    listed = listed_shards(job)
    assert [p.name for p in listed] == ["shard_000.jsonl"]
    n = merge_jsonl_shards(listed, job / "results.jsonl")
    assert n == 1
    assert "99" not in (job / "results.jsonl").read_text()


def test_dry_run_orchestrator(tmp_path: Path, capsys):
    from cue_training.evaluation.rollouts.orchestrator import Orchestrator

    raw = json.loads(_tiny_config(tmp_path).read_text())
    raw["benchmarks"]["tau2"]["enabled"] = False
    (tmp_path / "rollouts.json").write_text(json.dumps(raw), encoding="utf-8")
    cfg = load_rollouts_config(tmp_path / "rollouts.json", repo_root=ROOT)
    orch = Orchestrator(
        cfg,
        rollout_type="baseline",
        methods=["base"],
        dry_run=True,
        limit=1,
        run_id="dry1",
    )
    result = orch.run()
    assert result["run_id"] == "dry1"
    assert result["summaries"] == []
    out = capsys.readouterr().out
    assert "serve_groups" in out
    assert "jobs" in out
    # No side-effect job trees under the dry-run output root.
    assert not any((tmp_path / "out" / "dry1").rglob("results.jsonl"))
