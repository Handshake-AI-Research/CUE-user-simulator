"""Regression: a sibling vLLM failing to start must not kill healthy owned servers."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

from cue_training.evaluation.rollouts import vllm as vllm_mod
from cue_training.evaluation.rollouts.planning import VllmRequest
from cue_training.evaluation.rollouts.vllm import OwnedServer, VllmSupervisor


def _req(name: str, port: int, gpu: int) -> VllmRequest:
    return VllmRequest(
        name=name,
        model=f"model-{name}",
        served_name=name,
        port=port,
        gpus=(gpu,),
        tensor_parallel=1,
        gpu_memory_utilization=0.9,
        max_model_len=None,
    )


def _supervisor(tmp_path: Path) -> VllmSupervisor:
    script = tmp_path / "start_vllm.sh"
    script.write_text("#!/bin/bash\nsleep 1\n")
    cfg = SimpleNamespace(
        resources=SimpleNamespace(
            start_vllm_script=str(script),
            vllm_venv=None,
            cache_root=None,
            vllm_python="3.12",
        )
    )
    return VllmSupervisor(cfg=cfg, log_dir=tmp_path / "logs")  # type: ignore[arg-type]


def test_failed_start_keeps_other_owned_servers(tmp_path: Path, monkeypatch) -> None:
    sup = _supervisor(tmp_path)

    # A healthy USP server already running (e.g. mid-SimArena).
    usp = OwnedServer(
        request=_req("usp", 8205, 2),
        base_url=sup.base_url(8205),
        pid=155719,
        log_path=tmp_path / "usp.log",
    )
    sup.owned.append(usp)

    stopped: list[int] = []
    monkeypatch.setattr(sup, "_stop_server", lambda server, *, reason: stopped.append(server.pid))
    monkeypatch.setattr(sup, "_wait_ready", lambda *a, **k: False)
    monkeypatch.setattr(
        vllm_mod.subprocess,
        "Popen",
        lambda *a, **k: SimpleNamespace(pid=900900),
    )

    failed = False
    try:
        sup._start(_req("sim:llama", 8100, 3), wait_timeout=1.0)
    except RuntimeError:
        failed = True

    assert failed, "a vLLM that never becomes ready must raise"
    # Only the just-started llama is torn down; USP survives.
    assert stopped == [900900]
    assert [s.pid for s in sup.owned] == [155719]
