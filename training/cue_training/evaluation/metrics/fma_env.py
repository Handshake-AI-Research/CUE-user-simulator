"""Join optional Environment Error tags into metrics Success F1."""

from __future__ import annotations

from pathlib import Path

from cue_training.evaluation.common.io import read_jsonl
from cue_training.evaluation.fma.paths import tagged_path
from cue_training.evaluation.fma.sim_errors import ENVIRONMENT_ERROR_LABEL


def fma_source_id(*, kind: str, method: str, simulator: str, arm: str) -> str:
    """Build the FMA ``source_id`` for a scored candidate job."""

    arm_s = str(arm or "as_is")
    method_s = str(method or "")
    sim_s = str(simulator or "")
    kind_s = str(kind or "baseline")
    if kind_s == "cue" or method_s in {"general", "refined", "proposals"}:
        return f"cue-{method_s}-{sim_s}|{arm_s}"
    if method_s == "base":
        return f"base-{sim_s}|{arm_s}"
    return f"{method_s}-{sim_s}|{arm_s}"


def load_env_error_episode_ids(
    fma_run: Path | str | None,
    *,
    source_id: str | None = None,
) -> set[str]:
    """Episode ids tagged ``Environment Error`` in an FMA run (optionally one source)."""

    if not fma_run:
        return set()
    path = tagged_path(Path(fma_run))
    if not path.is_file():
        return set()
    out: set[str] = set()
    for row in read_jsonl(str(path)):
        if str(row.get("label") or "").strip() != ENVIRONMENT_ERROR_LABEL:
            continue
        if source_id is not None and str(row.get("source_id") or "") != source_id:
            continue
        eid = str(row.get("episode_id") or "").strip()
        if eid:
            out.add(eid)
    return out


def env_error_ids_for_job(
    fma_run: Path | str | None,
    *,
    kind: str,
    method: str,
    simulator: str,
    arm: str = "as_is",
) -> set[str]:
    if not fma_run:
        return set()
    return load_env_error_episode_ids(
        fma_run,
        source_id=fma_source_id(kind=kind, method=method, simulator=simulator, arm=arm),
    )
