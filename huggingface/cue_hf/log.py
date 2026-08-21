"""Minimal logging helpers."""

from __future__ import annotations

import sys
import time


def log(tag: str, msg: str) -> None:
    ts = time.strftime("%H:%M:%S")
    print(f"[{ts}] [{tag}] {msg}", file=sys.stderr, flush=True)


def warn(tag: str, msg: str) -> None:
    ts = time.strftime("%H:%M:%S")
    print(f"[{ts}] [{tag}] WARN: {msg}", file=sys.stderr, flush=True)
