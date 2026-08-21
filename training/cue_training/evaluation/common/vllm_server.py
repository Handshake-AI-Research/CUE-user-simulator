"""Launch + health-check a vLLM OpenAI-compatible server for local user-sim models.

vLLM does continuous batching, so one GPU serves many concurrent user-turn generations,
removing the sidecar's single-stream (one-at-a-time) bottleneck for plain-LM variants
(``base_local``). Only plain autoregressive LMs benefit: the CUE soft/hard decoders use
custom soft-prefix ``inputs_embeds`` / command decoding that vLLM can't serve, so they
stay on the HF path in the sidecar.

Usage (on a GPU node; install vLLM there first with ``uv pip install vllm`` -- it is not a
declared project dep because it is CUDA/Linux-only and would break cross-platform sync):
    python -m cue_training.evaluation.common.vllm_server --model meta-llama/Llama-3.1-8B-Instruct \
        --port 8000 --gpus 0
Then point the sidecar at it: ``--vllm_base_url http://127.0.0.1:8000/v1``. The rollout
processes' ``base_local`` turns are served (batched) by vLLM via litellm's ``hosted_vllm``
provider. This module only shells out to the ``vllm`` CLI + does an HTTP health check, so
it imports without vLLM installed.
"""

from __future__ import annotations

import argparse
import os
import subprocess
import time
import urllib.request

from cue_training.runlog.log import log, warn


def server_command(
    model: str, *, host: str, port: int, gpu_memory_utilization: float = 0.9,
    max_model_len: int | None = None, extra: list[str] | None = None,
) -> list[str]:
    cmd = ["vllm", "serve", model, "--host", host, "--port", str(port),
           "--gpu-memory-utilization", str(gpu_memory_utilization)]
    if max_model_len:
        cmd += ["--max-model-len", str(max_model_len)]
    return cmd + list(extra or [])


def base_url(host: str, port: int) -> str:
    # localhost when serving on 0.0.0.0
    reachable = "127.0.0.1" if host in ("0.0.0.0", "") else host
    return f"http://{reachable}:{port}/v1"


def wait_ready(url: str, *, timeout: float = 1800.0, interval: float = 5.0) -> bool:
    """Poll the OpenAI ``/models`` endpoint until the server is up (model loaded)."""

    models_url = url.rstrip("/") + "/models"
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            with urllib.request.urlopen(models_url, timeout=10) as resp:  # noqa: S310
                if resp.status == 200:
                    return True
        except Exception:  # noqa: BLE001
            pass
        time.sleep(interval)
    return False


def launch(
    model: str, *, host: str = "0.0.0.0", port: int = 8000, gpus: str | None = None,
    gpu_memory_utilization: float = 0.9, max_model_len: int | None = None,
    extra: list[str] | None = None,
) -> subprocess.Popen:
    env = dict(os.environ)
    if gpus is not None:
        env["CUDA_VISIBLE_DEVICES"] = str(gpus)
    cmd = server_command(model, host=host, port=port,
                         gpu_memory_utilization=gpu_memory_utilization,
                         max_model_len=max_model_len, extra=extra)
    log("vllm", f"launching: {' '.join(cmd)} (CUDA_VISIBLE_DEVICES={gpus})")
    return subprocess.Popen(cmd, env=env)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Launch a vLLM OpenAI server for user-sim models.")
    p.add_argument("--model", required=True)
    p.add_argument("--host", default="0.0.0.0")
    p.add_argument("--port", type=int, default=8000)
    p.add_argument("--gpus", default=None, help="CUDA_VISIBLE_DEVICES for this server (e.g. '0').")
    p.add_argument("--gpu_memory_utilization", type=float, default=0.9)
    p.add_argument("--max_model_len", type=int, default=None)
    p.add_argument("--wait_timeout", type=float, default=1800.0)
    p.add_argument("extra", nargs="*", help="Extra args passed through to `vllm serve`.")
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    proc = launch(args.model, host=args.host, port=args.port, gpus=args.gpus,
                  gpu_memory_utilization=args.gpu_memory_utilization,
                  max_model_len=args.max_model_len, extra=args.extra)
    url = base_url(args.host, args.port)
    if wait_ready(url, timeout=args.wait_timeout):
        log("vllm", f"ready at {url} (served model: {args.model})")
    else:
        warn("vllm", f"not ready after {args.wait_timeout}s; check the vllm log")
    proc.wait()  # keep the server in the foreground


if __name__ == "__main__":
    main()
