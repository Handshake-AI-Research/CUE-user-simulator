"""Managed evaluation.common.sidecar lifecycle with port allocation and cleanup."""

from __future__ import annotations

import os
import signal
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from cue_training.runlog.log import log, warn


def physical_device(gpu: int) -> str:
    """Map a config GPU index through an inherited ``CUDA_VISIBLE_DEVICES`` mask.

    A seed worker launched with ``CUDA_VISIBLE_DEVICES=<physical>`` numbers its own
    devices from 0, but the value handed to a child is absolute, so writing the logical
    index would drag every worker's sidecar back onto physical GPU 0.
    """

    mask = os.environ.get("CUDA_VISIBLE_DEVICES", "").strip()
    if not mask:
        return str(gpu)
    devices = [item.strip() for item in mask.split(",") if item.strip()]
    if 0 <= gpu < len(devices):
        return devices[gpu]
    warn("rollouts.sidecar", f"gpu={gpu} outside CUDA_VISIBLE_DEVICES={mask}; using as physical id")
    return str(gpu)


@dataclass
class SidecarHandle:
    url: str
    port: int
    pid: int
    log_path: Path
    owned: bool = True


@dataclass
class SidecarSupervisor:
    log_dir: Path
    python: str = field(default_factory=lambda: sys.executable)
    owned: list[SidecarHandle] = field(default_factory=list)
    dry_run: bool = False
    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False)

    def health(self, url: str, *, timeout: float = 2.0) -> bool:
        # GET /health is a model-free liveness probe (never builds a GPU sim), so it stays fast
        # and won't spam next_turn errors while the sidecar is busy decoding. Only a 200 counts:
        # a foreign server on the port answers 404 here but then 404s /decode_manuals too.
        try:
            req = urllib.request.Request(url.rstrip("/") + "/health", method="GET")
            with urllib.request.urlopen(req, timeout=timeout) as resp:  # noqa: S310
                return resp.status == 200
        except urllib.error.HTTPError:
            return False
        except (urllib.error.URLError, TimeoutError, OSError):
            return False

    def _pids_on_port(self, port: int) -> list[int]:
        """Pids listening on ``port``. Prefer ``ss`` (present on most clusters); fall back to lsof/fuser."""

        import re

        for cmd in (
            ["ss", "-lptn", f"sport = :{port}"],
            ["lsof", "-t", f"-iTCP:{port}", "-sTCP:LISTEN"],
            ["fuser", f"{port}/tcp"],
        ):
            try:
                out = subprocess.run(cmd, capture_output=True, text=True, timeout=10)  # noqa: S603
            except (FileNotFoundError, subprocess.SubprocessError):
                continue
            text = f"{out.stdout}\n{out.stderr}"
            if cmd[0] == "ss":
                pids = [int(m) for m in re.findall(r"pid=(\d+)", text)]
            else:
                pids = [
                    int(p)
                    for p in text.replace(f"{port}/tcp:", "").split()
                    if p.strip().isdigit()
                ]
            # Preserve order, drop duplicates.
            seen: set[int] = set()
            ordered: list[int] = []
            for pid in pids:
                if pid not in seen:
                    seen.add(pid)
                    ordered.append(pid)
            if ordered:
                return ordered
        return []

    def _kill_port(self, port: int) -> None:
        """SIGKILL whatever is listening on ``port`` and WAIT until the port is actually free.

        A single kill can race a survivor that is slow to release the socket (or a supervisor
        respawning it), letting a fresh start silently reuse a process running stale code -- the
        exact trap that produced empty-manual runs. Poll + re-kill until the port is clear.
        """

        for _ in range(5):
            pids = self._pids_on_port(port)
            if not pids:
                return
            for pid in pids:
                try:
                    os.kill(pid, signal.SIGKILL)
                    log("rollouts.sidecar", f"killed stale process pid={pid} on port {port}")
                except OSError:
                    pass
            time.sleep(1.0)  # let the OS release the port before re-checking
        if self._pids_on_port(port):
            warn("rollouts.sidecar", f"port {port} still occupied after kill attempts; fresh start may reuse stale code")

    def ensure(self, port: int, args: list[str], *, host: str = "127.0.0.1", gpu: int | None = None,
               fresh: bool = False) -> str:
        with self._lock:
            return self._ensure_locked(port, args, host=host, gpu=gpu, fresh=fresh)

    def _ensure_locked(self, port: int, args: list[str], *, host: str = "127.0.0.1", gpu: int | None = None,
                       fresh: bool = False) -> str:
        url = f"http://{host}:{port}"
        already_started = any(h.port == port for h in self.owned)
        if fresh and not already_started:
            # Force a brand-new sidecar: kill anything on the port and confirm it's gone before
            # starting, so we never reuse a survivor running stale code (source of empty-manual
            # runs). Do NOT fall through to the health-reuse branch below.
            self._kill_port(port)
            if self.health(url):  # a survivor is still answering -> kill again, don't reuse it
                self._kill_port(port)
        elif self.health(url):
            log("rollouts.sidecar", f"reusing sidecar at {url}")
            return url
        if self.dry_run:
            log("rollouts.sidecar", f"dry-run: would start sidecar on {url}")
            return url
        self.log_dir.mkdir(parents=True, exist_ok=True)
        log_path = self.log_dir / f"sidecar_{port}.log"
        cmd = [
            self.python,
            "-m",
            "cue_training.evaluation.common.sidecar",
            "--host",
            host,
            "--port",
            str(port),
            *args,
        ]
        # Pin the decoder to a specific GPU so it doesn't load onto GPU 0 alongside the sim vLLM
        # (which OOMs). CUDA_VISIBLE_DEVICES=g makes the sidecar's "cuda" == that physical GPU.
        env = dict(os.environ)
        physical = None
        if gpu is not None:
            physical = physical_device(gpu)
            env["CUDA_VISIBLE_DEVICES"] = physical
        log(
            "rollouts.sidecar",
            f"starting on gpu={physical if physical is not None else 'inherit'}: {' '.join(cmd)}",
        )
        handle = log_path.open("w", encoding="utf-8")
        proc = subprocess.Popen(  # noqa: S603
            cmd,
            env=env,
            stdout=handle,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
        self.owned.append(SidecarHandle(url=url, port=port, pid=proc.pid, log_path=log_path))
        deadline = time.time() + 120
        while time.time() < deadline:
            # Check our own process first: if it died (e.g. the port was taken and bind failed),
            # a stranger answering on that port must not be mistaken for a ready sidecar.
            if proc.poll() is not None:
                squatters = self._pids_on_port(port)
                detail = f"; port {port} held by pid(s) {squatters}" if squatters else ""
                raise RuntimeError(f"sidecar exited before ready; see {log_path}{detail}")
            if self.health(url):
                log("rollouts.sidecar", f"ready at {url}")
                return url
            time.sleep(0.5)
        self.stop_owned()
        raise RuntimeError(f"sidecar not ready at {url}; see {log_path}")

    def stop_owned(self) -> None:
        for handle in list(self.owned):
            if not handle.owned:
                continue
            try:
                os.killpg(handle.pid, signal.SIGTERM)
            except OSError:
                try:
                    os.kill(handle.pid, signal.SIGTERM)
                except OSError:
                    pass
            log("rollouts.sidecar", f"stopped owned sidecar pid={handle.pid}")
        self.owned.clear()


def sidecar_args_for_job(
    *,
    variant: str,
    sim_model: str,
    sim_api_base: str | None,
    sim_api_key_env: str,
    decoder_dir: str | None = None,
    baseline_output_dir: str | None = None,
    baseline_artifacts_dir: str | None = None,
    example_retrieval: dict[str, Any] | None = None,
    vllm_urls: dict[str, str] | None = None,
    usp_vllm_model: str = "wangkevin02/USP",
) -> list[str]:
    args = [
        "--sim_model", str(sim_model),
        "--sim_api_key_env", str(sim_api_key_env),
    ]
    if sim_api_base:
        args += ["--sim_api_base", str(sim_api_base)]
    if decoder_dir:
        args += ["--decoder_dir", str(decoder_dir)]
    if baseline_output_dir:
        args += ["--baseline_output_dir", str(baseline_output_dir)]
    if baseline_artifacts_dir:
        args += ["--baseline_artifacts_dir", str(baseline_artifacts_dir)]
    urls = vllm_urls or {}
    if "sim" in urls:
        args += ["--vllm_base_url", urls["sim"]]
    if "userlm" in urls:
        args += ["--userlm_vllm_base_url", urls["userlm"], "--userlm_vllm_model", "userlm"]
    if "usp" in urls:
        args += ["--usp_vllm_base_url", urls["usp"], "--usp_vllm_model", usp_vllm_model]
    if example_retrieval:
        from cue_training.evaluation.common.user_sims import example_retrieval_cli_flags

        args += example_retrieval_cli_flags(
            enabled=bool(example_retrieval.get("enabled")),
            k_sessions=int(example_retrieval.get("k_sessions") or 8),
            n_general=int(example_retrieval.get("n_general") or 2),
            n_specific=int(example_retrieval.get("n_specific") or 2),
        )
    _ = variant  # reserved for future per-variant flags
    return args
