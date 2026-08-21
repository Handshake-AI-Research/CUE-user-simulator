"""Managed vLLM lifecycle: start/reuse/cleanup without importing vLLM into the main process."""

from __future__ import annotations

import json
import os
import signal
import subprocess
import threading
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from cue_training.evaluation.rollouts.config import RolloutsConfig
from cue_training.evaluation.rollouts.planning import VllmRequest
from cue_training.evaluation.rollouts.sidecar import physical_device
from cue_training.runlog.log import log, warn


@dataclass
class OwnedServer:
    request: VllmRequest
    base_url: str
    pid: int
    log_path: Path
    owned: bool = True


@dataclass
class VllmSupervisor:
    cfg: RolloutsConfig
    log_dir: Path
    owned: list[OwnedServer] = field(default_factory=list)
    dry_run: bool = False
    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False)

    def base_url(self, port: int) -> str:
        return f"http://127.0.0.1:{port}/v1"

    def models(self, base_url: str, *, timeout: float = 5.0) -> list[str] | None:
        url = base_url.rstrip("/") + "/models"
        try:
            with urllib.request.urlopen(url, timeout=timeout) as resp:  # noqa: S310
                if resp.status != 200:
                    return None
                payload = json.loads(resp.read().decode("utf-8"))
        except (urllib.error.URLError, TimeoutError, json.JSONDecodeError, OSError):
            return None
        data = payload.get("data") or []
        return [str(item.get("id") or "") for item in data if isinstance(item, dict)]

    def compatible(self, req: VllmRequest) -> bool:
        names = self.models(self.base_url(req.port))
        if names is None:
            return False
        wanted = {req.served_name, req.model, Path(str(req.model)).name}
        return any(n in wanted for n in names if n)

    def ensure(self, req: VllmRequest, *, wait_timeout: float = 1800.0) -> str:
        with self._lock:
            url = self.base_url(req.port)
            if self.compatible(req):
                log("rollouts.vllm", f"reusing compatible server at {url} for {req.name}")
                return url
            existing = self.models(url)
            if existing is not None:
                raise RuntimeError(
                    f"port {req.port} already serves incompatible models {existing}; "
                    f"needed {req.served_name!r} ({req.model})"
                )
            if self.dry_run:
                log("rollouts.vllm", f"dry-run: would start {req.name} on {url}")
                return url
            return self._start(req, wait_timeout=wait_timeout)

    def _start(self, req: VllmRequest, *, wait_timeout: float) -> str:
        script = self.cfg.resources.start_vllm_script
        if script is None or not Path(script).is_file():
            raise RuntimeError(f"start_vllm script missing: {script}")
        self.log_dir.mkdir(parents=True, exist_ok=True)
        log_path = self.log_dir / f"vllm_{req.name.replace(':', '_')}_{req.port}.log"
        # Resolve logical -> physical here and drop the inherited mask, so a seed worker's
        # server lands on that worker's GPU even if start_vllm.sh does not remap.
        gpus = ",".join(physical_device(int(g)) for g in req.gpus)
        env = dict(os.environ)
        env.pop("CUDA_VISIBLE_DEVICES", None)
        env.update(
            {
                "MODEL": str(req.model),
                "PORT": str(req.port),
                "GPUS": gpus,
                "TP": str(req.tensor_parallel),
                "GPU_MEM_UTIL": str(req.gpu_memory_utilization),
                "SERVED_MODEL_NAME": req.served_name,
                "ENABLE_PROMPT_EMBEDS": "1" if req.prompt_embeds else "0",
                "INSTALL": env.get("INSTALL", "0"),
                "WAIT_TIMEOUT": str(int(wait_timeout)),
            }
        )
        if req.max_model_len is not None:
            env["MAX_MODEL_LEN"] = str(req.max_model_len)
        if self.cfg.resources.vllm_venv is not None:
            env["VLLM_VENV"] = str(self.cfg.resources.vllm_venv)
        if self.cfg.resources.cache_root is not None:
            env["CACHE_ROOT"] = str(self.cfg.resources.cache_root)
        env["PYVER"] = self.cfg.resources.vllm_python
        log("rollouts.vllm", f"starting {req.name}: model={req.model} port={req.port} gpus={gpus}")
        handle = log_path.open("w", encoding="utf-8")
        proc = subprocess.Popen(  # noqa: S603
            ["bash", str(script)],
            env=env,
            stdout=handle,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
        self.owned.append(
            OwnedServer(request=req, base_url=self.base_url(req.port), pid=proc.pid, log_path=log_path)
        )
        if not self._wait_ready(req, proc.pid, wait_timeout=wait_timeout):
            # Only tear down the server that failed — do not nuke unrelated owned
            # servers (e.g. a live USP mid-SimArena when a sibling vLLM fails ready).
            failed = [s for s in self.owned if s.pid == proc.pid]
            self.owned = [s for s in self.owned if s.pid != proc.pid]
            for server in failed:
                self._stop_server(server, reason=f"failed ready {req.name}")
            raise RuntimeError(f"vLLM {req.name} failed to become ready; see {log_path}")
        return self.base_url(req.port)

    def _wait_ready(self, req: VllmRequest, pid: int, *, wait_timeout: float) -> bool:
        deadline = time.time() + wait_timeout
        while time.time() < deadline:
            if self.compatible(req):
                log("rollouts.vllm", f"{req.name} ready at {self.base_url(req.port)}")
                return True
            try:
                os.kill(pid, 0)
            except OSError:
                warn("rollouts.vllm", f"{req.name} exited before ready (pid={pid})")
                return False
            time.sleep(5)
        warn("rollouts.vllm", f"{req.name} timed out after {wait_timeout}s")
        return False

    def _stop_server(self, server: OwnedServer, *, reason: str) -> None:
        if not server.owned:
            return
        try:
            os.killpg(server.pid, signal.SIGTERM)
        except OSError:
            try:
                os.kill(server.pid, signal.SIGTERM)
            except OSError:
                pass
        # Wait for exit so the next wave's vLLM does not see stale VRAM occupancy.
        deadline = time.time() + 90.0
        while time.time() < deadline:
            try:
                os.kill(server.pid, 0)
            except OSError:
                break
            time.sleep(0.5)
        else:
            try:
                os.killpg(server.pid, signal.SIGKILL)
            except OSError:
                try:
                    os.kill(server.pid, signal.SIGKILL)
                except OSError:
                    pass
            warn("rollouts.vllm", f"force-killed pid={server.pid} ({server.request.name})")
        log("rollouts.vllm", f"{reason} pid={server.pid} ({server.request.name})")

    def stop_owned(self) -> None:
        for server in list(self.owned):
            self._stop_server(server, reason="stopped owned server")
        self.owned.clear()

    def release_not_needed(self, needed: list[VllmRequest]) -> None:
        """Stop owned servers that are not an exact match for something in ``needed``.

        Job waves run sequentially (e.g. userlm then usp). Without this, the prior
        method's policy vLLM keeps ~half the seed GPU and the next one OOMs.
        """

        keep = {(r.model, r.served_name, r.port, tuple(r.gpus)) for r in needed}
        remaining: list[OwnedServer] = []
        for server in list(self.owned):
            req = server.request
            key = (req.model, req.served_name, req.port, tuple(req.gpus))
            if key in keep:
                remaining.append(server)
                continue
            self._stop_server(
                server,
                reason=f"released {req.name} (port={req.port}) before next wave",
            )
        self.owned = remaining

    def ensure_group(self, group: list[VllmRequest]) -> dict[str, str]:
        urls: dict[str, str] = {}
        for req in group:
            urls[req.name] = self.ensure(req)
        return urls
