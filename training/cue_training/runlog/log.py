"""One shared logging + progress helper for cue/baselines/evaluation.

Goals: a single, consistent stdout format (``[tag] message``, always flushed),
uniform rank-gating for distributed runs, and progress bars that stay readable in
redirected log files. This module is intentionally dependency-light -- stdlib only,
with ``tqdm`` imported lazily inside :func:`progress` and no ``torch`` import -- so it
is cheap and safe to import from the eval sidecar and other non-GPU paths.

Typical use::

    from cue_training.runlog.log import log, warn, error, step, stage, phase, progress

    log("decoder", "starting training")
    with phase("decoder", "load session embeddings"):
        embeddings = load_session_embeddings(path)
    for batch in progress(loader, desc="decoder epoch 1", unit="batch"):
        ...
    step("decoder", global_step, {"loss": loss})
"""

from __future__ import annotations

import os
import sys
import time
from contextlib import contextmanager
from typing import Any, Iterable, Iterator

# Fixed float format so metric lines look identical across encoder/soft/hard/baselines.
_FLOAT_FMT = ".4f"
# How often the non-TTY progress heartbeat is allowed to print (seconds).
_HEARTBEAT_SECONDS = 30.0


def rank(explicit: int | None = None) -> int:
    """Current process rank.

    Returns ``explicit`` when given (callers that already computed their rank via
    ``torch.distributed`` pass it so gating never diverges), else reads the ``RANK``
    env var set by torchrun (default 0). Never imports torch.
    """

    if explicit is not None:
        return explicit
    try:
        return int(os.environ.get("RANK", "0"))
    except ValueError:
        return 0


def _world_size() -> int:
    try:
        return int(os.environ.get("WORLD_SIZE", "1"))
    except ValueError:
        return 1


def _emit(text: str) -> None:
    print(text, flush=True)


def _gated(rank_zero: bool, rank_: int | None) -> bool:
    """Return True when the current process should emit output."""

    if not rank_zero:
        return True
    return rank(rank_) == 0


def log(tag: str, msg: str, *, rank: int | None = None, rank_zero: bool = True) -> None:
    """Print ``[tag] msg`` (flushed). Rank-0 only by default under distributed runs."""

    if _gated(rank_zero, rank):
        _emit(f"[{tag}] {msg}")


def _rank_suffix(rank_: int | None) -> str:
    """`` [rankN]`` context for warnings/errors when running distributed."""

    if _world_size() > 1:
        return f" [rank{rank(rank_)}]"
    return ""


def warn(tag: str, msg: str, *, rank: int | None = None) -> None:
    """Print a warning on every rank (rank-local failures must not disappear)."""

    _emit(f"[{tag}] WARN:{_rank_suffix(rank)} {msg}")


def error(tag: str, msg: str, *, rank: int | None = None) -> None:
    """Print an error on every rank (rank-local failures must not disappear)."""

    _emit(f"[{tag}] ERROR:{_rank_suffix(rank)} {msg}")


def _fmt_value(value: Any) -> str:
    if isinstance(value, float):
        return format(value, _FLOAT_FMT)
    return str(value)


def step(
    tag: str,
    step: int,
    metrics: dict[str, Any],
    *,
    rank: int | None = None,
    rank_zero: bool = True,
) -> None:
    """Print a canonical ``[tag] step {n} | k: v | ...`` metric line (printed mirror only)."""

    if not _gated(rank_zero, rank):
        return
    body = " | ".join(f"{key}: {_fmt_value(value)}" for key, value in metrics.items())
    suffix = f" | {body}" if body else ""
    _emit(f"[{tag}] step {step}{suffix}")


def stage(
    tag: str,
    msg: str,
    *,
    start: float | None = None,
    rank: int | None = None,
    rank_zero: bool = True,
) -> None:
    """One-line status; appends ``(Xs)`` elapsed when ``start`` (a perf_counter) is given."""

    if not _gated(rank_zero, rank):
        return
    if start is None:
        _emit(f"[{tag}] {msg}")
    else:
        _emit(f"[{tag}] {msg} ({time.perf_counter() - start:.1f}s)")


@contextmanager
def phase(
    tag: str,
    name: str,
    *,
    rank: int | None = None,
    rank_zero: bool = True,
) -> Iterator[None]:
    """Log ``name ...`` on enter and ``name done (Xs)`` on exit (``FAILED`` on error).

    The primary tool for making otherwise-silent long steps visibly progress. Gate it to
    match rank-guarded work (e.g. a rank-0-only model load) so other ranks do not print
    "done" for work they skipped.
    """

    emit = _gated(rank_zero, rank)
    started = time.perf_counter()
    if emit:
        _emit(f"[{tag}] {name} ...")
    try:
        yield
    except BaseException:
        if emit:
            _emit(f"[{tag}] {name} FAILED ({time.perf_counter() - started:.1f}s)")
        raise
    if emit:
        _emit(f"[{tag}] {name} done ({time.perf_counter() - started:.1f}s)")


class _HeartbeatBar:
    """Minimal drop-in for a tqdm bar that prints throttled newline updates.

    Used when writing to a non-TTY (redirected log file / pipe): a live tqdm bar would
    spam carriage-return fragments, so instead we emit an occasional greppable line via
    :func:`log` and stay silent otherwise. Supports the subset of the tqdm API the CUE
    call sites use (iteration, ``update``, ``set_postfix``, ``close``, context manager).
    """

    def __init__(self, tag: str, iterable: Iterable | None, total: int | None, desc: str, unit: str) -> None:
        self._tag = tag
        self._iterable = iterable
        self._total = total
        self._desc = desc or "progress"
        self._unit = unit or "it"
        self._n = 0
        self._start = time.perf_counter()
        self._last = 0.0
        self._postfix = ""

    def _line(self) -> str:
        elapsed = time.perf_counter() - self._start
        rate = self._n / elapsed if elapsed > 0 else 0.0
        if self._total:
            pct = 100.0 * self._n / self._total
            head = f"{self._desc}: {self._n}/{self._total} ({pct:.0f}%)"
        else:
            head = f"{self._desc}: {self._n}"
        line = f"{head} {rate:.1f}{self._unit}/s elapsed {elapsed:.0f}s"
        return f"{line} | {self._postfix}" if self._postfix else line

    def _maybe_emit(self, *, force: bool = False) -> None:
        now = time.perf_counter()
        if force or now - self._last >= _HEARTBEAT_SECONDS:
            self._last = now
            log(self._tag, self._line())

    def __iter__(self) -> Iterator[Any]:
        assert self._iterable is not None
        for item in self._iterable:
            yield item
            self._n += 1
            self._maybe_emit()
        self._maybe_emit(force=True)

    def update(self, n: int = 1) -> None:
        self._n += n
        self._maybe_emit()

    def set_postfix(self, *args: Any, **kwargs: Any) -> None:
        parts = []
        for arg in args:
            if isinstance(arg, dict):
                parts.extend(f"{k}={_fmt_value(v)}" for k, v in arg.items())
        parts.extend(f"{k}={_fmt_value(v)}" for k, v in kwargs.items())
        self._postfix = ", ".join(parts)

    def set_description(self, desc: str) -> None:
        self._desc = desc

    def close(self) -> None:
        self._maybe_emit(force=True)

    def __enter__(self) -> "_HeartbeatBar":
        return self

    def __exit__(self, *exc: Any) -> None:
        self.close()


class _NullBar:
    """No-op bar for gated-off ranks: supports the tqdm subset but prints nothing."""

    def __init__(self, iterable: Iterable | None) -> None:
        self._iterable = iterable

    def __iter__(self) -> Iterator[Any]:
        return iter(self._iterable or ())

    def update(self, n: int = 1) -> None:  # noqa: ARG002
        pass

    def set_postfix(self, *args: Any, **kwargs: Any) -> None:
        pass

    def set_description(self, desc: str) -> None:  # noqa: ARG002
        pass

    def close(self) -> None:
        pass

    def __enter__(self) -> "_NullBar":
        return self

    def __exit__(self, *exc: Any) -> None:
        pass


def progress(
    iterable: Iterable | None = None,
    *,
    total: int | None = None,
    desc: str,
    unit: str = "it",
    rank: int | None = None,
    rank_zero: bool = True,
    leave: bool = False,
    disable: bool | None = None,
    tag: str | None = None,
    **kwargs: Any,
) -> Any:
    """Consistent progress bar for every CUE loop.

    - ``total`` may be passed explicitly so streaming / no-``__len__`` iterables get an ETA
      without materializing (never calls ``len()`` when ``total`` is given). Also supports
      the counter style (``bar = progress(total=T, ...)`` then ``bar.update()``).
    - Rank-gated by default; pass ``rank_zero=False`` (or an explicit ``disable``) for
      per-rank bars.
    - TTY-aware: a live ``tqdm`` bar on a terminal; on a redirected non-TTY it degrades to a
      throttled newline heartbeat via :func:`log` so ``gpu*.log`` / ``nohup`` stay clean.
    """

    if disable is None:
        disable = rank_zero and rank_fn(rank) != 0
    if disable:
        return _NullBar(iterable)

    heartbeat_tag = tag or desc or "progress"
    if not sys.stderr.isatty():
        return _HeartbeatBar(heartbeat_tag, iterable, total, desc, unit)

    from tqdm.auto import tqdm

    return tqdm(
        iterable,
        total=total,
        desc=desc,
        unit=unit,
        leave=leave,
        dynamic_ncols=True,
        **kwargs,
    )


# Alias so ``progress`` can call rank() without shadowing its own ``rank`` kwarg.
rank_fn = rank
