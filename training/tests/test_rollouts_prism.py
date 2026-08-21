"""Network-free PRISM closed-loop fixture tests."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from cue_training.evaluation.common.user_sims import UserSim
from cue_training.evaluation.rollouts.artifacts import JobSpec, RunPaths
from cue_training.evaluation.rollouts.backends.base import BackendContext
from cue_training.evaluation.rollouts.backends.closed_loop import (
    count_user_turns,
    default_turn_budget,
    opening_prompt,
    prism_user_system_prompt,
    run_closed_loop,
)
from cue_training.evaluation.rollouts.backends.prism import (
    PrismBackend,
    _episode_id,
    _is_terminal_result,
    _result_key,
    join_prism_embeddings,
    prepare_resume,
)
from cue_training.evaluation.rollouts.config import load_rollouts_config


ROOT = Path(__file__).resolve().parents[1]
FIXTURE = ROOT / "tests/fixtures/prism/normalized.jsonl"


class _ScriptedSim(UserSim):
    def __init__(self, turns: list[str]):
        self._turns = list(turns)
        self._i = 0

    def next_turn(self, history):
        if self._i >= len(self._turns):
            return "", True
        text = self._turns[self._i]
        self._i += 1
        return text, self._i >= len(self._turns)


def test_turn_budget_from_metadata():
    row = json.loads(FIXTURE.read_text(encoding="utf-8").splitlines()[0])
    assert count_user_turns(row["real_conversation"]) == 2
    assert default_turn_budget(row) == 2
    assert default_turn_budget(row, cap=1) == 1


def test_closed_loop_roles_and_schema():
    row = json.loads(FIXTURE.read_text(encoding="utf-8").splitlines()[0])
    sim = _ScriptedSim(["u1", "u2", "u3"])
    replies = ["a1", "a2", "a3"]
    idx = {"i": 0}

    def assistant(conversation, system):
        assert system
        assert conversation[-1]["role"] == "user"
        text = replies[idx["i"]]
        idx["i"] += 1
        return text

    record = run_closed_loop(
        row=row,
        user_sim=sim,
        assistant_reply=assistant,
        assistant_system="sys",
        max_user_turns=2,
        provenance={"source_harness": "test", "arm": "as_is", "variant": "base_api"},
    )
    assert record["dataset"] == "prism"
    assert record["episode_id"] == "fix0"
    roles = [t["role"] for t in record["rollout_conversation"]]
    assert roles == ["user", "assistant", "user", "assistant"]
    assert record["metadata"]["source_harness"] == "test"
    assert record["task_description"]


def test_closed_loop_seeds_real_opening_prompt():
    row = json.loads(FIXTURE.read_text(encoding="utf-8").splitlines()[0])
    assert opening_prompt(row) == "Hello there"
    sim = _ScriptedSim(["sim turn 2", "sim turn 3"])
    replies = ["a1", "a2", "a3"]
    idx = {"i": 0}

    def assistant(conversation, system):
        text = replies[idx["i"]]
        idx["i"] += 1
        return text

    record = run_closed_loop(
        row=row,
        user_sim=sim,
        assistant_reply=assistant,
        assistant_system="sys",
        max_user_turns=2,
        provenance={"source_harness": "test", "arm": "as_is", "variant": "base_api"},
        opening=opening_prompt(row),
    )
    conv = record["rollout_conversation"]
    user_contents = [t["content"] for t in conv if t["role"] == "user"]
    # First user turn is the real opening prompt verbatim; the sim drives the rest.
    assert user_contents[0] == "Hello there"
    assert user_contents[1] == "sim turn 2"


def test_prism_prompt_primes_condition_and_preserves_persona_slot():
    row = {
        "task_description": "Discuss a difficult social issue",
        "metadata": {"domain": "diversity", "conversation_type": "controversy_guided"},
    }
    prompt = prism_user_system_prompt(row)
    assert "Controversy-guided" in prompt
    assert "Do not restart or restate the opening" in prompt
    assert "Persona and behavioral conditioning" in prompt
    assert "must not replace the established topic" in prompt


def test_prism_backend_dry_run(tmp_path: Path):
    cfg_path = tmp_path / "cfg.json"
    cfg_path.write_text(
        json.dumps(
            {
                "benchmarks": {
                    "tau2": {"enabled": False},
                    "simulatorarena": {},
                    "prism": {"enabled": True, "normalized": str(FIXTURE)},
                },
                "simulators": [{"name": "gpt", "kind": "api", "model": "gpt-4o-mini"}],
                "assistant": {"model": "gpt-4o-mini"},
                "resources": {"gpus": [0], "output_root": str(tmp_path / "out")},
                "cue": {},
                "baselines": {"artifact_root": str(tmp_path / "b")},
                "metrics": {},
            }
        ),
        encoding="utf-8",
    )
    cfg = load_rollouts_config(cfg_path, repo_root=ROOT)
    job = JobSpec(
        rollout_type="baseline",
        method="base",
        benchmark="prism",
        domain="diversity",
        simulator="gpt",
        arms=["as_is"],
        variant="base_api",
    )
    paths = RunPaths(run_id="t", run_dir=tmp_path / "out" / "t", config_hash="x")
    ctx = BackendContext(
        cfg=cfg,
        paths=paths,
        job=job,
        job_dir=tmp_path / "job",
        dry_run=True,
        limit=1,
    )
    backend = PrismBackend()
    prep = backend.prepare(ctx)
    assert prep["n_rows"] == 1
    info = backend.run(ctx, prep)
    assert info["n_planned"] == 1


def test_prism_backend_injects_persona(tmp_path: Path, monkeypatch):
    """personas.jsonl rows keyed by (variant, arm, episode_id) are passed into build_episode_sim."""

    cfg_path = tmp_path / "cfg.json"
    cfg_path.write_text(
        json.dumps(
            {
                "benchmarks": {
                    "tau2": {"enabled": False},
                    "simulatorarena": {},
                    "prism": {"enabled": True, "normalized": str(FIXTURE)},
                },
                "simulators": [{"name": "gpt", "kind": "api", "model": "gpt-4o-mini"}],
                "assistant": {"model": "gpt-4o-mini"},
                "resources": {"gpus": [0], "output_root": str(tmp_path / "out"), "concurrency": 1},
                "cue": {},
                "baselines": {"artifact_root": str(tmp_path / "b")},
                "metrics": {},
            }
        ),
        encoding="utf-8",
    )
    cfg = load_rollouts_config(cfg_path, repo_root=ROOT)
    job = JobSpec(
        rollout_type="cue",
        method="proposals",
        benchmark="prism",
        domain="diversity",
        simulator="gpt",
        arms=["paired"],
        variant="baseline:realusersim",
    )
    job_dir = tmp_path / "job"
    job_dir.mkdir()
    persona = {"commands": [{"text": "be brief"}], "manual_text": "be brief"}
    (job_dir / "personas.jsonl").write_text(
        json.dumps(
            {
                "variant": "baseline:realusersim",
                "arm": "paired",
                "episode_id": "fix0",
                "persona": persona,
            }
        )
        + "\n",
        encoding="utf-8",
    )
    paths = RunPaths(run_id="t", run_dir=tmp_path / "out" / "t", config_hash="x")
    ctx = BackendContext(
        cfg=cfg,
        paths=paths,
        job=job,
        job_dir=job_dir,
        dry_run=False,
        limit=1,
        work_dir=tmp_path / "work",
    )
    captured: dict[str, Any] = {}

    class _StubSim(UserSim):
        def next_turn(self, history):
            return "hi", True

    def _fake_build(
        *, variant, arm, row, persona=None, cue_embedding=None, extra=None, system_prompt=None
    ):
        captured["persona"] = persona
        captured["episode_id"] = row.get("episode_id")
        captured["cue_embedding"] = cue_embedding
        captured["system_prompt"] = system_prompt
        return _StubSim()

    monkeypatch.setattr(
        "cue_training.evaluation.rollouts.backends.prism.build_episode_sim", _fake_build
    )
    monkeypatch.setattr(
        "cue_training.evaluation.rollouts.backends.prism.PrismBackend._assistant_fn",
        lambda self, ctx, tasks=None: (lambda row: (lambda conversation, system: "ok")),
    )
    backend = PrismBackend()
    prep = backend.prepare(ctx)
    backend.run(ctx, prep)
    assert captured["episode_id"] == "fix0"
    assert captured["persona"] == persona
    assert captured["cue_embedding"] is None  # proposals does not require CUE vectors
    assert "PRISM conversation continuation" in captured["system_prompt"]


def test_join_prism_embeddings_paired_and_shuffled(tmp_path: Path):
    rows = [
        {"episode_id": "c0", "metadata": {"domain": "diversity"}},
        {"episode_id": "c1", "metadata": {"domain": "diversity"}},
    ]
    emb_path = tmp_path / "emb.jsonl"
    emb_path.write_text(
        "\n".join(
            [
                json.dumps(
                    {
                        "session_id": "c0",
                        "trajectory_embedding": [1.0, 0.0],
                        "metadata": {"episode_id": "c0", "domain": "diversity"},
                    }
                ),
                json.dumps(
                    {
                        "session_id": "c1",
                        "trajectory_embedding": [0.0, 1.0],
                        "metadata": {"episode_id": "c1", "domain": "diversity"},
                    }
                ),
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    tasks, counts = join_prism_embeddings(
        rows,
        emb_path,
        arms=["paired", "sample_shuffled"],
        require_embeddings=True,
        seed=0,
    )
    assert counts["n_resolvable"] == 2
    assert len(tasks) == 4
    paired = {_episode_id(r): emb for r, arm, emb in tasks if arm == "paired"}
    assert paired["c0"] == pytest.approx([1.0, 0.0])
    assert paired["c1"] == pytest.approx([0.0, 1.0])
    shuffled = [emb for _, arm, emb in tasks if arm == "sample_shuffled"]
    assert all(emb is not None and len(emb) == 2 for emb in shuffled)


def test_prism_backend_passes_cue_embedding_for_decoder(tmp_path: Path, monkeypatch):
    emb_path = tmp_path / "prism_emb.jsonl"
    emb_path.write_text(
        json.dumps(
            {
                "session_id": "fix0",
                "trajectory_embedding": [3.0, 4.0],
                "metadata": {"episode_id": "fix0", "domain": "diversity"},
            }
        )
        + "\n",
        encoding="utf-8",
    )
    cfg_path = tmp_path / "cfg.json"
    cfg_path.write_text(
        json.dumps(
            {
                "benchmarks": {
                    "tau2": {"enabled": False},
                    "simulatorarena": {},
                    "prism": {
                        "enabled": True,
                        "normalized": str(FIXTURE),
                        "embeddings": str(emb_path),
                    },
                },
                "simulators": [{"name": "gpt", "kind": "api", "model": "gpt-4o-mini"}],
                "assistant": {"model": "gpt-4o-mini"},
                "resources": {"gpus": [0], "output_root": str(tmp_path / "out"), "concurrency": 1},
                "cue": {"joint_training_checkpoint": str(tmp_path / "ckpt")},
                "baselines": {"artifact_root": str(tmp_path / "b")},
                "metrics": {},
            }
        ),
        encoding="utf-8",
    )
    cfg = load_rollouts_config(cfg_path, repo_root=ROOT)
    job = JobSpec(
        rollout_type="cue",
        method="general",
        benchmark="prism",
        domain="diversity",
        simulator="gpt",
        arms=["paired"],
        variant="decoder",
        needs_cue_embeddings=True,
        needs_sidecar=True,
    )
    paths = RunPaths(run_id="t", run_dir=tmp_path / "out" / "t", config_hash="x")
    ctx = BackendContext(
        cfg=cfg,
        paths=paths,
        job=job,
        job_dir=tmp_path / "job",
        dry_run=False,
        limit=1,
        work_dir=tmp_path / "work",
    )
    (tmp_path / "job").mkdir()
    captured: dict[str, Any] = {}

    class _StubSim(UserSim):
        def next_turn(self, history):
            return "hi", True

    def _fake_build(
        *, variant, arm, row, persona=None, cue_embedding=None, extra=None, system_prompt=None
    ):
        captured["cue_embedding"] = cue_embedding
        captured["arm"] = arm
        return _StubSim()

    monkeypatch.setattr(
        "cue_training.evaluation.rollouts.backends.prism.build_episode_sim", _fake_build
    )
    monkeypatch.setattr(
        "cue_training.evaluation.rollouts.backends.prism.PrismBackend._assistant_fn",
        lambda self, ctx, tasks=None: (lambda row: (lambda conversation, system: "ok")),
    )
    monkeypatch.setattr(
        "cue_training.evaluation.rollouts.backends.prism.PrismBackend._predecode_manuals",
        lambda self, ctx, tasks, extra: {},
    )
    backend = PrismBackend()
    prep = backend.prepare(ctx)
    assert prep["embeddings"] == str(emb_path)
    backend.run(ctx, prep)
    assert captured["arm"] == "paired"
    assert captured["cue_embedding"] is not None
    assert len(captured["cue_embedding"]) == 2
    assert captured["cue_embedding"] == pytest.approx([3.0, 4.0])


def test_is_terminal_result_legacy_and_status() -> None:
    assert _is_terminal_result({"episode_id": "a", "arm": "as_is"})  # legacy ok
    assert _is_terminal_result({"status": "ok", "episode_id": "a"})
    assert not _is_terminal_result({"status": "error", "error": "boom"})
    assert not _is_terminal_result({"episode_id": "a", "error": "boom"})


def test_prepare_resume_compacts_errors_and_keeps_ok(tmp_path: Path) -> None:
    results = tmp_path / "results.jsonl"
    rollouts = tmp_path / "rollout.prism.jsonl"
    rows = [
        {"variant": "base_api", "arm": "as_is", "episode_id": "a", "status": "ok"},
        {"variant": "base_api", "arm": "as_is", "episode_id": "b", "status": "error", "error": "x"},
        {"variant": "base_api", "arm": "as_is", "episode_id": "c"},  # legacy ok
    ]
    results.write_text("\n".join(json.dumps(r) for r in rows) + "\n", encoding="utf-8")
    rollouts.write_text(
        "\n".join(
            [
                json.dumps({
                    "episode_id": "a",
                    "metadata": {"variant": "base_api", "arm": "as_is"},
                    "rollout_conversation": [{"role": "user", "content": "hi"}],
                }),
                json.dumps({
                    "episode_id": "orphan",
                    "metadata": {"variant": "base_api", "arm": "as_is"},
                    "rollout_conversation": [{"role": "user", "content": "no"}],
                }),
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    done = prepare_resume(tmp_path, default_variant="base_api")
    assert done == {
        _result_key({"variant": "base_api", "arm": "as_is", "episode_id": "a"}),
        _result_key({"variant": "base_api", "arm": "as_is", "episode_id": "c"}),
    }
    kept = [json.loads(line) for line in results.read_text(encoding="utf-8").splitlines() if line]
    assert [r["episode_id"] for r in kept] == ["a", "c"]
    kept_roll = [
        json.loads(line) for line in rollouts.read_text(encoding="utf-8").splitlines() if line
    ]
    assert [r["episode_id"] for r in kept_roll] == ["a"]


def test_prism_run_resumes_completed_episodes(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    cfg_path = tmp_path / "cfg.json"
    cfg_path.write_text(
        json.dumps(
            {
                "root": str(ROOT),
                "benchmarks": {
                    "prism": {
                        "enabled": True,
                        "normalized": str(FIXTURE),
                    }
                },
                "simulators": [{"name": "gpt", "kind": "api", "model": "gpt-4o-mini"}],
                "assistant": {"model": "gpt-4o-mini"},
                "resources": {"gpus": [0], "output_root": str(tmp_path / "out"), "concurrency": 1},
                "cue": {},
                "baselines": {"artifact_root": str(tmp_path / "b")},
                "metrics": {},
            }
        ),
        encoding="utf-8",
    )
    cfg = load_rollouts_config(cfg_path, repo_root=ROOT)
    job = JobSpec(
        rollout_type="baseline",
        method="base",
        benchmark="prism",
        domain="diversity",
        simulator="gpt",
        arms=["as_is"],
        variant="base_api",
    )
    job_dir = tmp_path / "job"
    job_dir.mkdir()
    # Seed one completed episode matching the first fixture id.
    first = json.loads(FIXTURE.read_text(encoding="utf-8").splitlines()[0])
    eid = first.get("episode_id") or first.get("task_id")
    (job_dir / "results.jsonl").write_text(
        json.dumps(
            {
                "episode_id": eid,
                "variant": "base_api",
                "arm": "as_is",
                "status": "ok",
                "n_user_turns": 1,
            }
        )
        + "\n",
        encoding="utf-8",
    )
    (job_dir / "rollout.prism.jsonl").write_text(
        json.dumps(
            {
                "episode_id": eid,
                "metadata": {"variant": "base_api", "arm": "as_is"},
                "rollout_conversation": [{"role": "user", "content": "prior"}],
            }
        )
        + "\n",
        encoding="utf-8",
    )
    paths = RunPaths(run_id="t", run_dir=tmp_path / "out" / "t", config_hash="x")
    ctx = BackendContext(
        cfg=cfg,
        paths=paths,
        job=job,
        job_dir=job_dir,
        dry_run=False,
        limit=2,
        work_dir=tmp_path / "work",
    )
    seen: list[str] = []

    class _StubSim(UserSim):
        def next_turn(self, history):
            return "hi", True

    def _fake_build(
        *, variant, arm, row, persona=None, cue_embedding=None, extra=None, system_prompt=None
    ):
        seen.append(str(row.get("episode_id") or row.get("task_id")))
        return _StubSim()

    monkeypatch.setattr("cue_training.evaluation.rollouts.backends.prism.build_episode_sim", _fake_build)
    monkeypatch.setattr(
        "cue_training.evaluation.rollouts.backends.prism.PrismBackend._assistant_fn",
        lambda self, ctx, tasks=None: (lambda row: (lambda conversation, system: "ok")),
    )
    backend = PrismBackend()
    prep = backend.prepare(ctx)
    backend.run(ctx, prep)
    assert eid not in seen
    assert len(seen) == 1
    results = [
        json.loads(line)
        for line in (job_dir / "results.jsonl").read_text(encoding="utf-8").splitlines()
        if line
    ]
    assert len(results) == 2
    assert {r["episode_id"] for r in results} >= {eid, seen[0]}
