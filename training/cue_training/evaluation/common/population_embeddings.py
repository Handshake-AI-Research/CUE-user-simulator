"""Auto-export the paired evaluation population for sampled CUE arms."""

from __future__ import annotations

import os
import threading
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator

from cue_training.evaluation.common.io import read_jsonl

_EXPORT_LOCK = threading.Lock()
_EXPORTED_THIS_RUN: set[str] = set()


@contextmanager
def _interprocess_lock(path: Path) -> Iterator[None]:
    """Serialize population exports across seed workers sharing one embeddings path."""

    path.parent.mkdir(parents=True, exist_ok=True)
    lock_path = path.with_name(f".{path.name}.lock")
    fd = os.open(str(lock_path), os.O_CREAT | os.O_RDWR, 0o644)
    try:
        import fcntl

        fcntl.flock(fd, fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(fd, fcntl.LOCK_UN)
    finally:
        os.close(fd)


def _mtime_or_none(path: Path) -> float | None:
    try:
        st = path.stat()
    except FileNotFoundError:
        return None
    return st.st_mtime if st.st_size > 0 else None


def ensure_population_embeddings(
    cfg: Any,
    job: Any,
    *,
    dry_run: bool = False,
    force: bool = False,
) -> Path | None:
    """Create the configured benchmark embedding file from its paired conversations."""

    from cue_training.evaluation.common.sampler_arms import has_sampled_arms

    if not has_sampled_arms(job.arms):
        return None
    checkpoint = cfg.cue.joint_training_checkpoint or cfg.cue.decoder_dir
    if checkpoint is None:
        raise ValueError("sampled CUE arms require cue.joint_training_checkpoint/decoder_dir")

    output, records = _population_source(cfg, job)
    if output is None:
        raise ValueError(
            f"{job.benchmark} sampled CUE arms require a configured benchmark embeddings path"
        )
    output = Path(output)
    try:
        key = str(output.resolve())
    except FileNotFoundError:
        key = str(output)
    if not force and _mtime_or_none(output) is not None:
        return output
    if dry_run:
        return output

    mtime_before = _mtime_or_none(output)

    # In-process simulators prepare concurrently; seed workers are separate processes
    # that share the same embeddings path under CUE_STORAGE_ROOT.
    with _EXPORT_LOCK:
        if key in _EXPORTED_THIS_RUN and _mtime_or_none(output) is not None:
            return output
        with _interprocess_lock(output):
            if key in _EXPORTED_THIS_RUN and _mtime_or_none(output) is not None:
                return output
            mtime_now = _mtime_or_none(output)
            if mtime_now is not None and not force:
                _EXPORTED_THIS_RUN.add(key)
                return output
            # Sibling finished while we blocked on the flock (force or cold start).
            if mtime_now is not None and (mtime_before is None or mtime_now > mtime_before):
                _EXPORTED_THIS_RUN.add(key)
                return output
            from cue_training.infer.export import export_embedding_records
            from cue_training.runlog.log import log

            gpu = cfg.resources.sidecar_gpu
            device = f"cuda:{gpu}" if gpu is not None else "cuda"
            log(
                "rollouts",
                f"paired population embeddings missing; encoding {len(records)} rows -> {output}",
            )
            export_embedding_records(
                checkpoint,
                records,
                output,
                device=device,
                batch_size=cfg.cue.decode_batch_size,
                session_preprocess=cfg.cue.session_preprocess_for(job.domain),
            )
            _EXPORTED_THIS_RUN.add(key)
    return output


def _population_source(cfg: Any, job: Any) -> tuple[Path | None, list[dict[str, Any]]]:
    if job.benchmark == "tau2":
        bench = cfg.benchmarks.tau2
        return bench.embeddings, read_jsonl(str(bench.normalized))
    if job.benchmark == "prism":
        bench = cfg.benchmarks.prism
        return bench.embeddings, read_jsonl(str(bench.normalized))
    if job.benchmark == "simulatorarena":
        bench = cfg.benchmarks.simulatorarena[job.domain]
        from cue_training.evaluation.simulatorarena.data import load_annotations, to_normalized_rows

        annotations = load_annotations(bench.annotations)
        rows = to_normalized_rows(annotations, job.domain, bench.background)
        return bench.embeddings, rows
    raise ValueError(f"unsupported sampled CUE benchmark: {job.benchmark}")
