from __future__ import annotations

import hashlib
import json
import pickle
import time
from pathlib import Path
from typing import Any

from cue_training.utils.config import storage_root


def file_hash(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()[:16]


def paths_hash(paths: list[Path]) -> str:
    """Stable combined content hash over one or more baseline JSONLs."""

    h = hashlib.sha256()
    for path in sorted(Path(p).resolve() for p in paths):
        h.update(str(path).encode("utf-8"))
        h.update(b"\0")
        h.update(file_hash(path).encode("utf-8"))
        h.update(b"\0")
    return h.hexdigest()[:16]


def default_cache_dir(baseline: Path) -> Path:
    stem = baseline.stem.replace(".", "_")
    return storage_root() / "outputs" / "metrics" / stem / "state"


def load_manifest(cache_dir: Path) -> dict[str, Any] | None:
    path = cache_dir / "manifest.json"
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def write_manifest(
    cache_dir: Path,
    *,
    baseline: Path | None = None,
    baselines: list[Path] | None = None,
    config: dict[str, Any],
) -> dict[str, Any]:
    cache_dir.mkdir(parents=True, exist_ok=True)
    paths = [Path(p) for p in (baselines or ([baseline] if baseline is not None else []))]
    if not paths:
        raise ValueError("write_manifest requires baseline or baselines")
    primary = paths[0]
    manifest = {
        "baseline": str(primary),
        "baselines": [str(p) for p in paths],
        "baseline_hash": paths_hash(paths) if len(paths) > 1 else file_hash(primary),
        "config": config,
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    (cache_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest


def manifest_matches(
    cache_dir: Path,
    *,
    baseline: Path | None = None,
    baselines: list[Path] | None = None,
    config: dict[str, Any],
) -> bool:
    """True when baseline content hash matches and stored config is compatible with ``config``.

    Compatibility ignores the ``metrics`` list and allows the stored config to be a *superset*
    of the requested one (so ``--metrics sim2real_self_similarity --merge`` can reuse a prior
    full fit that also recorded OSS/authenticity keys).

    Pass ``baselines`` (multi-path aggregate fit) to match against the combined content hash;
    otherwise a single ``baseline`` path is hashed as before.
    """

    paths = [Path(p) for p in (baselines or ([baseline] if baseline is not None else []))]
    if not paths:
        return False
    expected = paths_hash(paths) if len(paths) > 1 else file_hash(paths[0])
    manifest = load_manifest(cache_dir)
    if not manifest or manifest.get("baseline_hash") != expected:
        return False
    stored = {k: v for k, v in (manifest.get("config") or {}).items() if k != "metrics"}
    requested = {k: v for k, v in config.items() if k != "metrics"}
    return all(stored.get(k) == v for k, v in requested.items())


def save_pickle(cache_dir: Path, name: str, obj: Any) -> None:
    cache_dir.mkdir(parents=True, exist_ok=True)
    with (cache_dir / name).open("wb") as handle:
        pickle.dump(obj, handle)


def load_pickle(cache_dir: Path, name: str) -> Any | None:
    path = cache_dir / name
    if not path.exists():
        return None
    with path.open("rb") as handle:
        return pickle.load(handle)

