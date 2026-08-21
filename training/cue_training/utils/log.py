"""Minimal logging helpers."""

from __future__ import annotations

import sys
from datetime import datetime


def log(tag: str, msg: str, *, rank: int = 0) -> None:
    if rank != 0:
        return
    ts = datetime.now().strftime("%H:%M:%S")
    print(f"[{ts}] [{tag}] {msg}", file=sys.stderr, flush=True)


def warn(tag: str, msg: str, *, rank: int = 0) -> None:
    if rank != 0:
        return
    ts = datetime.now().strftime("%H:%M:%S")
    print(f"[{ts}] [{tag}] WARN: {msg}", file=sys.stderr, flush=True)
