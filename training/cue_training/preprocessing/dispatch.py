"""Least-inflight load balancer for local vLLM replicas."""

from __future__ import annotations

import threading
from contextlib import contextmanager
from typing import Iterator

# Bake-off / data-annotation default ports (see scripts/start_data_annotation_vllms.sh):
#   8000 Llama rollouts + profile contrast
#   8001 Qwen  profile contrast (+ optional Qwen rollouts)
#   8901/8902 reserved for OSS metrics (not routed here)
DEFAULT_LLAMA_PORTS = [8000]
DEFAULT_QWEN_PORTS = [8001]


class LeastInflightPool:
    """Pick the replica with the fewest in-flight requests."""

    def __init__(self, bases: list[str]) -> None:
        if not bases:
            raise ValueError("LeastInflightPool requires at least one base URL")
        self._bases = list(bases)
        self._inflight = {b: 0 for b in self._bases}
        self._lock = threading.Lock()

    @contextmanager
    def acquire(self) -> Iterator[str]:
        with self._lock:
            base = min(self._bases, key=lambda b: self._inflight[b])
            self._inflight[base] += 1
        try:
            yield base
        finally:
            with self._lock:
                self._inflight[base] -= 1


class ReplicaRouter:
    """Map model name substrings to least-inflight replica pools."""

    def __init__(self, pools: dict[str, LeastInflightPool] | None = None) -> None:
        self._pools = pools or {}

    @contextmanager
    def api_base_for(self, model: str, fallback: str | None = None) -> Iterator[str | None]:
        for key, pool in self._pools.items():
            if key in model:
                with pool.acquire() as base:
                    yield base
                    return
        yield fallback


def build_default_router(
    llama_ports: list[int] | None = None,
    qwen_ports: list[int] | None = None,
    host: str = "127.0.0.1",
) -> ReplicaRouter:
    llama_ports = llama_ports or list(DEFAULT_LLAMA_PORTS)
    qwen_ports = qwen_ports or list(DEFAULT_QWEN_PORTS)
    return ReplicaRouter(
        {
            "Llama": LeastInflightPool([f"http://{host}:{p}/v1" for p in llama_ports]),
            "Qwen": LeastInflightPool([f"http://{host}:{p}/v1" for p in qwen_ports]),
        }
    )
