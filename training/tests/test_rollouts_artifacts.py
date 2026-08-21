"""Rollout artifact plumbing: arm-replace merges, PPOL persona pools, vLLM reuse."""

from __future__ import annotations

import json
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path

from cue_training.evaluation.rollouts.artifacts import merge_jsonl_replace_arms, promote_final_artifacts, row_arm
from cue_training.evaluation.rollouts.config import SimulatorConfig, load_rollouts_config
from cue_training.evaluation.rollouts.planning import VllmRequest
from cue_training.evaluation.rollouts.ppol_artifacts import (
    PPOL_TRAIN_DOMAIN,
    assign_personas_from_pool,
    load_persona_pool,
    persona_pool_path,
    ppol_artifacts_dir,
    ppol_best_program_path,
    ppol_sim_model,
    sidecar_covers_as_is,
    update_persona_pool_from_sidecar,
    write_persona_pool,
)
from cue_training.evaluation.rollouts.vllm import VllmSupervisor


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.write_text(
        "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows),
        encoding="utf-8",
    )


def test_merge_jsonl_replace_arms(tmp_path: Path):
    existing = tmp_path / "results.jsonl"
    new = tmp_path / "new.jsonl"
    _write_jsonl(
        existing,
        [
            {"arm": "paired", "episode_id": "e1", "v": 1},
            {"arm": "sampled_pop", "episode_id": "e1", "v": 1},
            {"arm": "sampled_pop", "episode_id": "e2", "v": 1},
            {"metadata": {"arm": "sample_shuffled"}, "episode_id": "e1", "v": 1},
        ],
    )
    _write_jsonl(
        new,
        [
            {"arm": "sampled_pop", "episode_id": "e1", "v": 2},
            {"arm": "sampled_pop", "episode_id": "e2", "v": 2},
        ],
    )
    n = merge_jsonl_replace_arms(existing, new, ["sampled_pop"])
    rows = [json.loads(line) for line in existing.read_text(encoding="utf-8").splitlines()]
    assert n == 4
    by_arm = {}
    for r in rows:
        by_arm.setdefault(row_arm(r), []).append(r)
    assert len(by_arm["paired"]) == 1 and by_arm["paired"][0]["v"] == 1
    assert len(by_arm["sample_shuffled"]) == 1
    assert all(r["v"] == 2 for r in by_arm["sampled_pop"])
    assert len(by_arm["sampled_pop"]) == 2


def test_promote_merges_when_replace_arms(tmp_path: Path):
    job = tmp_path / "job"
    stage = tmp_path / "stage"
    job.mkdir()
    stage.mkdir()
    _write_jsonl(
        job / "results.jsonl",
        [
            {"arm": "paired", "episode_id": "e1"},
            {"arm": "sampled_pop", "episode_id": "e1", "old": True},
        ],
    )
    _write_jsonl(stage / "results.jsonl", [{"arm": "sampled_pop", "episode_id": "e1", "old": False}])
    promote_final_artifacts(stage, job, ["results.jsonl"], replace_arms=["sampled_pop"])
    rows = [json.loads(line) for line in (job / "results.jsonl").read_text().splitlines()]
    assert len(rows) == 2
    pop = [r for r in rows if r["arm"] == "sampled_pop"][0]
    assert pop["old"] is False
    assert any(r["arm"] == "paired" for r in rows)


def test_ppol_sim_model_api_and_vllm():
    api = SimulatorConfig(name="gpt", kind="api", model="gpt-5.4-mini")
    assert ppol_sim_model(api) == "gpt-5.4-mini"
    vllm = SimulatorConfig(
        name="llama",
        kind="vllm",
        model="meta-llama/Llama-3.1-8B-Instruct",
        served_name="meta-llama/Llama-3.1-8B-Instruct",
    )
    assert ppol_sim_model(vllm) == "hosted_vllm/meta-llama/Llama-3.1-8B-Instruct"


def test_ppol_artifacts_dir_per_sim(tmp_path: Path):
    gpt = ppol_artifacts_dir(tmp_path, "gpt-5.4-mini")
    llama = ppol_artifacts_dir(tmp_path, "hosted_vllm/meta-llama/Llama-3.1-8B-Instruct")
    assert gpt.name == f"{PPOL_TRAIN_DOMAIN}_gpt-5.4-mini"
    assert "hosted_vllm__meta-llama__Llama-3.1-8B-Instruct" in llama.name
    assert gpt != llama
    assert ppol_best_program_path(tmp_path, "gpt-5.4-mini") == gpt / "best_program.py"


def test_persona_pool_roundtrip_and_assign(tmp_path: Path):
    pool = tmp_path / "personas_pool.jsonl"
    write_persona_pool(pool, ["persona-a", "persona-b", "persona-a"])
    assert load_persona_pool(pool) == ["persona-a", "persona-b"]

    sidecar = tmp_path / "sidecar.jsonl"
    sidecar.write_text(
        '{"variant":"baseline:ppol","arm":"as_is","episode_id":"e1","persona":"persona-c"}\n',
        encoding="utf-8",
    )
    merged = update_persona_pool_from_sidecar(pool, sidecar)
    assert "persona-c" in merged
    assert len(merged) == 3

    rows = assign_personas_from_pool(
        ["w1", "w2", "w3", "w4"],
        merged,
        variant="baseline:ppol",
        seed=0,
    )
    assert len(rows) == 4
    assert all(r["arm"] == "as_is" for r in rows)
    assert {r["persona"] for r in rows} <= set(merged)
    assert sidecar_covers_as_is(sidecar, "baseline:ppol")
    assert not sidecar_covers_as_is(tmp_path / "missing.jsonl", "baseline:ppol")
    assert persona_pool_path(tmp_path) == tmp_path / "personas_pool.jsonl"


class _Handler(BaseHTTPRequestHandler):
    model_id = "llama"

    def do_GET(self):  # noqa: N802
        if self.path.rstrip("/").endswith("/models"):
            body = json.dumps({"data": [{"id": self.model_id}]}).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
            return
        self.send_response(404)
        self.end_headers()

    def log_message(self, *_args):
        return


def test_vllm_reuse_compatible(tmp_path: Path):
    server = HTTPServer(("127.0.0.1", 0), _Handler)
    port = server.server_address[1]
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        cfg_path = tmp_path / "cfg.json"
        cfg_path.write_text(
            json.dumps(
                {
                    "benchmarks": {"tau2": {"enabled": False}, "simulatorarena": {}, "prism": {"enabled": False}},
                    "simulators": [
                        {
                            "name": "llama",
                            "kind": "vllm",
                            "model": "meta-llama/Llama-3.1-8B-Instruct",
                            "served_name": "llama",
                            "gpus": [0],
                            "port": port,
                        }
                    ],
                    "assistant": {"model": "gpt-4o-mini"},
                    "resources": {"gpus": [0], "output_root": str(tmp_path / "out")},
                    "cue": {},
                    "baselines": {"artifact_root": str(tmp_path / "b")},
                    "metrics": {},
                }
            ),
            encoding="utf-8",
        )
        cfg = load_rollouts_config(cfg_path, repo_root=Path(__file__).resolve().parents[1])
        sup = VllmSupervisor(cfg, log_dir=tmp_path / "logs", dry_run=False)
        req = VllmRequest(
            name="sim:llama",
            model="meta-llama/Llama-3.1-8B-Instruct",
            served_name="llama",
            port=port,
            gpus=(0,),
            tensor_parallel=1,
            gpu_memory_utilization=0.9,
            max_model_len=4096,
        )
        url = sup.ensure(req)
        assert url.endswith(f":{port}/v1")
        assert not sup.owned  # reused, not owned
    finally:
        server.shutdown()
