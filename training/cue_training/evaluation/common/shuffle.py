"""Universal within-domain sample_shuffled mapping shared across methods."""

from __future__ import annotations

import random
from collections import defaultdict
from typing import Iterable, Mapping, Sequence

# Fixed seed so CUE, RealUserSim, USP, etc. share one derangement.
DEFAULT_SHUFFLE_SEED = 0


def build_shuffle_map(
    episode_ids: Sequence[str] | Iterable[str],
    *,
    domains: Mapping[str, str] | None = None,
    seed: int = DEFAULT_SHUFFLE_SEED,
) -> dict[str, str]:
    """Deterministic episode → other-episode map, shuffled **within domain**.

    Episodes are sorted before shuffling so encounter order does not matter: the
    same ``(ids, domains, seed)`` always yields the same mapping across prep,
    persona sidecars, and rollout runners.
    """

    ids = [str(eid) for eid in episode_ids if str(eid)]
    # Preserve first-seen while deduping.
    seen: set[str] = set()
    unique: list[str] = []
    for eid in ids:
        if eid in seen:
            continue
        seen.add(eid)
        unique.append(eid)

    by_domain: dict[str, list[str]] = defaultdict(list)
    for eid in unique:
        domain = "all"
        if domains is not None:
            domain = str(domains.get(eid) or "all")
        by_domain[domain].append(eid)

    out: dict[str, str] = {}
    for domain in sorted(by_domain):
        ordered = sorted(by_domain[domain])
        if not ordered:
            continue
        if len(ordered) == 1:
            out[ordered[0]] = ordered[0]
            continue
        rng = random.Random(f"{int(seed)}::{domain}")
        shuffled = ordered[:]
        # Rejection sample until we get a derangement when possible (n>=2).
        for _ in range(64):
            rng.shuffle(shuffled)
            if all(a != b for a, b in zip(ordered, shuffled, strict=True)):
                break
        else:
            # Rotate by one as a deterministic derangement fallback.
            shuffled = ordered[1:] + ordered[:1]
        for src, dst in zip(ordered, shuffled, strict=True):
            out[src] = dst
    return out


def domain_of(
    episode_id: str,
    *,
    explicit: str | None = None,
    record: Mapping[str, object] | None = None,
) -> str:
    """Best-effort domain label for grouping the universal shuffle."""

    if explicit:
        return str(explicit)
    if record:
        for key in ("tau2_domain", "domain"):
            val = record.get(key)
            if val:
                return str(val)
        meta = record.get("metadata")
        if isinstance(meta, Mapping):
            for key in ("tau2_domain", "domain"):
                val = meta.get(key)
                if val:
                    return str(val)
    eid = str(episode_id or "")
    if eid.startswith("airline"):
        return "airline"
    if eid.startswith("retail"):
        return "retail"
    return "all"
