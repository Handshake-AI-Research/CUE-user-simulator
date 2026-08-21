"""Collision-proof run/job paths, atomic manifests, resume, and shard merge."""

from __future__ import annotations

import json
import os
import tempfile
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Sequence


class ArtifactError(RuntimeError):
    """Raised for resume/hash/shard isolation failures."""


@dataclass
class JobSpec:
    rollout_type: str
    method: str
    benchmark: str
    domain: str
    simulator: str
    arms: list[str]
    variant: str
    needs_cue_embeddings: bool = False
    needs_sidecar: bool = False
    needs_policy_vllm: bool = False
    needs_sim_vllm: bool = False
    train: bool = False


@dataclass
class RunPaths:
    run_id: str
    run_dir: Path
    config_hash: str
    work_dir_root: Path | None = None

    def _job_subpath(self, job: JobSpec) -> Path:
        return (
            Path(job.benchmark)
            / job.domain
            / job.rollout_type
            / job.method
            / job.simulator
        )

    def job_dir(self, job: JobSpec) -> Path:
        """Main-outputs job dir (holds only final results + small metadata)."""

        return self.run_dir / self._job_subpath(job)

    def work_dir(self, job: JobSpec) -> Path:
        """Scratch job dir for bulky intermediates (cases, native runs, shards).

        Equals :meth:`job_dir` when no separate work root is configured.
        """

        base = self.work_dir_root or self.run_dir
        return base / self._job_subpath(job)


def default_run_id(config_hash: str, *, when: datetime | None = None) -> str:
    ts = (when or datetime.now(timezone.utc)).strftime("%Y%m%dT%H%M%SZ")
    return f"{ts}_{config_hash}"


def ensure_run_paths(
    output_root: Path,
    *,
    config_hash: str,
    run_id: str | None = None,
    resume: bool = False,
    work_root: Path | None = None,
    allow_hash_mismatch: bool = False,
) -> RunPaths:
    output_root.mkdir(parents=True, exist_ok=True)
    rid = run_id or default_run_id(config_hash)
    run_dir = output_root / rid
    marker = run_dir / "manifest.json"
    stored_hash = config_hash
    if resume:
        if not marker.is_file():
            raise ArtifactError(f"--resume requires existing run manifest at {marker}")
        existing = json.loads(marker.read_text(encoding="utf-8"))
        prev_hash = str(existing.get("config_hash") or "")
        if prev_hash != config_hash:
            if allow_hash_mismatch:
                # In-place arm reruns may tweak sampler knobs; keep the run's stored hash
                # so later resumes still match the original manifest.
                stored_hash = prev_hash or config_hash
            else:
                raise ArtifactError(
                    f"resume config hash mismatch: run={prev_hash} current={config_hash}"
                )
    else:
        run_dir.mkdir(parents=True, exist_ok=True)
    # Scratch root mirrors the run tree under a (usually bigger) disk; None -> same as run_dir.
    work_dir_root = (work_root / rid) if work_root is not None else None
    if work_dir_root is not None:
        work_dir_root.mkdir(parents=True, exist_ok=True)
    return RunPaths(
        run_id=rid, run_dir=run_dir, config_hash=stored_hash, work_dir_root=work_dir_root
    )


def promote_final_artifacts(
    work_dir: Path,
    job_dir: Path,
    names: list[str],
    *,
    replace_arms: Sequence[str] | None = None,
) -> None:
    """Copy (or arm-merge) named final artifacts from ``work_dir`` into ``job_dir``.

    When ``replace_arms`` is set and the destination already exists, rows for those
    arms are replaced in-place and other arms are kept. Callers that pass
    ``replace_arms`` should stage new arm-only files under a directory distinct from
    ``job_dir`` (e.g. ``scratch/.arm_rerun``).
    """

    job_dir.mkdir(parents=True, exist_ok=True)
    arms = tuple(str(a) for a in replace_arms) if replace_arms else None
    for name in names:
        src = work_dir / name
        dest = job_dir / name
        if not src.is_file():
            continue
        if src.resolve() == dest.resolve():
            continue
        if arms and dest.is_file():
            merge_jsonl_replace_arms(dest, src, arms, out_path=dest)
        else:
            dest.write_bytes(src.read_bytes())


def atomic_write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix=path.name + ".", dir=str(path.parent))
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            for row in rows:
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")
        os.replace(tmp, path)
    finally:
        if os.path.exists(tmp):
            try:
                os.unlink(tmp)
            except OSError:
                pass


def row_arm(row: dict[str, Any]) -> str:
    arm = row.get("arm")
    if arm:
        return str(arm)
    meta = row.get("metadata")
    if isinstance(meta, dict) and meta.get("arm"):
        return str(meta["arm"])
    return ""


def merge_jsonl_replace_arms(
    existing_path: Path,
    new_path: Path,
    replace_arms: Sequence[str],
    *,
    out_path: Path | None = None,
) -> int:
    """Keep non-replaced arms from ``existing_path``; swap in rows from ``new_path``.

    Returns the number of rows written.
    """

    replace = {str(a) for a in replace_arms}
    if not replace:
        raise ValueError("replace_arms must be nonempty")
    out_path = out_path or existing_path
    kept: list[dict[str, Any]] = []
    if existing_path.is_file():
        with existing_path.open(encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                row = json.loads(line)
                if row_arm(row) not in replace:
                    kept.append(row)
    added: list[dict[str, Any]] = []
    with new_path.open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            # Keep only the arms we intended to replace (ignore accidental extras).
            if row_arm(row) in replace:
                added.append(row)
    atomic_write_jsonl(out_path, kept + added)
    return len(kept) + len(added)


def atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix=path.name + ".", dir=str(path.parent))
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True)
            handle.write("\n")
        os.replace(tmp, path)
    finally:
        if os.path.exists(tmp):
            try:
                os.unlink(tmp)
            except OSError:
                pass


def write_run_manifest(paths: RunPaths, *, jobs: list[JobSpec], extra: dict[str, Any] | None = None) -> Path:
    payload = {
        "run_id": paths.run_id,
        "config_hash": paths.config_hash,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "jobs": [asdict(j) for j in jobs],
        **(extra or {}),
    }
    out = paths.run_dir / "manifest.json"
    atomic_write_json(out, payload)
    return out


def write_job_manifest(job_dir: Path, payload: dict[str, Any]) -> Path:
    out = job_dir / "manifest.json"
    atomic_write_json(out, payload)
    return out


def job_complete(job_dir: Path) -> bool:
    marker = job_dir / "COMPLETE"
    return marker.is_file() and (job_dir / "results.jsonl").is_file()


def mark_job_complete(job_dir: Path) -> None:
    (job_dir / "COMPLETE").write_text(f"{time.time()}\n", encoding="utf-8")


def listed_shards(job_dir: Path) -> list[Path]:
    manifest_path = job_dir / "manifest.json"
    if not manifest_path.is_file():
        return []
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    shards = manifest.get("shards") or []
    out: list[Path] = []
    for name in shards:
        path = job_dir / "shards" / str(name)
        if path.is_file():
            out.append(path)
    return out


def merge_jsonl_shards(shard_paths: list[Path], out_path: Path) -> int:
    """Merge only the provided shard list (no directory glob of stale shards)."""

    out_path.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with out_path.open("w", encoding="utf-8") as out:
        for shard in shard_paths:
            with shard.open(encoding="utf-8") as handle:
                for line in handle:
                    if line.strip():
                        out.write(line if line.endswith("\n") else line + "\n")
                        n += 1
    return n


def summarize_results(results_path: Path) -> dict[str, Any]:
    total = 0
    success = 0
    errors = 0
    if results_path.is_file():
        with results_path.open(encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                total += 1
                try:
                    row = json.loads(line)
                except json.JSONDecodeError:
                    errors += 1
                    continue
                if row.get("error"):
                    errors += 1
                reward = row.get("reward")
                if reward is True or reward == 1 or row.get("task_success") is True:
                    success += 1
                elif isinstance(reward, (int, float)) and float(reward) > 0:
                    success += 1
    return {
        "n_results": total,
        "n_success": success,
        "n_errors": errors,
        "success_rate": (success / total) if total else None,
    }
