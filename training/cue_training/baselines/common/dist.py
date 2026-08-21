"""Minimal torchrun-style distributed helpers (mirrors CUE's DDP pattern).

When a baseline step is launched with ``torchrun --nproc_per_node=N``, torchrun
sets ``RANK``/``WORLD_SIZE``/``LOCAL_RANK``. These helpers let the rollout driver
shard episodes across ranks (each on its own GPU) and let training gate its
single-process data-prep to rank 0 before the DDP-aware HF/TRL trainers run.
"""

from __future__ import annotations

import os
from datetime import timedelta
from pathlib import Path

_INITED = False


def get_dist_info() -> tuple[int, int, int]:
    return (
        int(os.environ.get("RANK", "0")),
        int(os.environ.get("WORLD_SIZE", "1")),
        int(os.environ.get("LOCAL_RANK", "0")),
    )


def is_distributed() -> bool:
    return get_dist_info()[1] > 1


def is_main_process() -> bool:
    return get_dist_info()[0] == 0


def resolve_device(device: str) -> str:
    """Bind each rank to its own GPU (``cuda:LOCAL_RANK``) when available."""

    _, _, local = get_dist_info()
    if device.startswith("cuda"):
        try:
            import torch

            if torch.cuda.is_available():
                return f"cuda:{local}"
        except Exception:  # noqa: BLE001
            return device
    return device


def maybe_init_process_group() -> bool:
    """Init a process group when running under torchrun; idempotent."""

    global _INITED
    if not is_distributed():
        return False
    import torch
    import torch.distributed as dist

    if dist.is_available() and not dist.is_initialized():
        rank, world, local = get_dist_info()
        backend = "nccl" if torch.cuda.is_available() else "gloo"
        if torch.cuda.is_available():
            torch.cuda.set_device(local)
        # Generous timeout: rank-0 data prep (LLM labeling) can run far longer than the
        # default 10-min NCCL watchdog while other ranks wait at the post-prep barrier.
        # Prefer running prep as a separate step (see scripts), but keep this as a guard.
        hours = float(os.environ.get("BASELINE_DIST_TIMEOUT_HOURS", "24"))
        dist.init_process_group(
            backend=backend, rank=rank, world_size=world, timeout=timedelta(hours=hours)
        )
        _INITED = True
    return dist.is_available() and dist.is_initialized()


def barrier() -> None:
    try:
        import torch.distributed as dist

        if dist.is_available() and dist.is_initialized():
            dist.barrier()
    except Exception:  # noqa: BLE001
        pass


def destroy() -> None:
    global _INITED
    try:
        import torch.distributed as dist

        if _INITED and dist.is_initialized():
            dist.destroy_process_group()
            _INITED = False
    except Exception:  # noqa: BLE001
        pass


def shard(items: list) -> list:
    """Return this rank's contiguous-stride slice of ``items``."""

    rank, world, _ = get_dist_info()
    if world <= 1:
        return items
    return items[rank::world]


def merge_jsonl_shards(shard_paths: list[Path], out_path: Path) -> None:
    with open(out_path, "w", encoding="utf-8") as out:
        for shard_path in shard_paths:
            if shard_path.exists():
                with open(shard_path, encoding="utf-8") as handle:
                    for line in handle:
                        out.write(line)
    for shard_path in shard_paths:
        try:
            shard_path.unlink()
        except FileNotFoundError:
            pass
