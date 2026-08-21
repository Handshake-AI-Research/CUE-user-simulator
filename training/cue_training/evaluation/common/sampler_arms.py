"""Diffusion-sampled CUE arms: unconditional and population-conditioned.

``sampled_uncond`` draws one CUE per episode with no set conditioner.
``sampled_pop`` draws one CUE per episode conditioned on a random subset of the
real eval population embeddings in that episode's domain (default size 8,
matching train-time CFG set size). Each episode gets its own random subset.
"""

from __future__ import annotations

import hashlib
from collections import defaultdict
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

SAMPLED_ARMS = ("sampled_uncond", "sampled_pop")
# Arms that need precomputed cue_embedding (cannot live-encode from trajectory).
PRECOMPUTED_CUE_ARMS = frozenset({"dataset_mean", *SAMPLED_ARMS})
# Match sampler train ``k`` (local CFG set size).
DEFAULT_POP_SET_K = 8


def has_sampled_arms(arms: Sequence[str] | None) -> bool:
    return bool(arms) and any(a in SAMPLED_ARMS for a in arms)


def needs_precomputed_embeddings(arms: Sequence[str] | None, *, recipe_needs: bool = False) -> bool:
    """True when the job must join/export real CUE vectors (not live-encode-only)."""

    if recipe_needs:
        return True
    return bool(arms) and any(a in PRECOMPUTED_CUE_ARMS for a in arms)


def sampler_cli_flags(
    *,
    checkpoint: str | Path | None,
    seed: int = 0,
    guidance_w: float = 1.5,
    ddim_steps: int | None = None,
    device: str | None = None,
    pop_max_k: int | None = None,
    project_manifold: bool | None = None,
) -> list[str]:
    """CLI flags forwarding sampler settings into tau2/simarena prep/run."""

    if checkpoint is None:
        return []
    flags = [
        f"--sampler_checkpoint={checkpoint}",
        f"--sampler_seed={int(seed)}",
        f"--sampler_guidance_w={float(guidance_w)}",
    ]
    if ddim_steps is not None:
        flags.append(f"--sampler_ddim_steps={int(ddim_steps)}")
    if device:
        flags.append(f"--sampler_device={device}")
    k = DEFAULT_POP_SET_K if pop_max_k is None else int(pop_max_k)
    flags.append(f"--sampler_pop_max_k={k}")
    if project_manifold is not None:
        flags.append(f"--sampler_project_manifold={'true' if project_manifold else 'false'}")
    return flags


def _parse_bool(value: Any) -> bool:
    return str(value).strip().lower() in {"1", "true", "yes", "y", "on"}


def add_sampler_argparse(parser: Any) -> None:
    """Register optional sampler flags on a prep/run ArgumentParser."""

    parser.add_argument("--sampler_checkpoint", default=None)
    parser.add_argument("--sampler_seed", type=int, default=0)
    parser.add_argument("--sampler_guidance_w", type=float, default=1.5)
    parser.add_argument("--sampler_ddim_steps", type=int, default=None)
    parser.add_argument("--sampler_device", default=None)
    parser.add_argument(
        "--sampler_pop_max_k",
        type=int,
        default=DEFAULT_POP_SET_K,
        help=(
            "Per-sample condition-set size for sampled_pop: each episode draws this "
            f"many random population embeddings (default {DEFAULT_POP_SET_K}, train k)."
        ),
    )
    parser.add_argument(
        "--sampler_project_manifold",
        type=_parse_bool,
        default=None,
        help=(
            "Project sampled vectors onto the decoder's LayerNorm manifold. Unset falls "
            "back to the checkpoint's train-time config; off-manifold vectors decode to "
            "token salad, so this should normally be true."
        ),
    )


def sampler_kwargs_from_args(args: Any) -> dict[str, Any]:
    """Pull sampler kwargs for ``build_cases`` / ``join_prism_embeddings`` (excludes shuffle seed)."""

    return {
        "sampler_checkpoint": getattr(args, "sampler_checkpoint", None),
        "sampler_seed": int(getattr(args, "sampler_seed", 0) or 0),
        "sampler_guidance_w": float(getattr(args, "sampler_guidance_w", 1.5) or 1.5),
        "sampler_ddim_steps": getattr(args, "sampler_ddim_steps", None),
        "sampler_device": getattr(args, "sampler_device", None),
        "sampler_pop_max_k": getattr(args, "sampler_pop_max_k", None),
        "sampler_project_manifold": getattr(args, "sampler_project_manifold", None),
    }


def _domain_seed(seed: int, domain: str, arm: str) -> int:
    digest = hashlib.sha1(f"{seed}:{domain}:{arm}".encode("utf-8")).hexdigest()
    return int(digest[:8], 16)


def _per_sample_pop_sets(
    pop: np.ndarray,
    n: int,
    k: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """Build ``[n, k_eff, D]`` — each row a random subset of ``pop`` (no replacement)."""

    if pop.ndim != 2 or pop.shape[0] == 0:
        raise ValueError(f"pop must be nonempty [P, D], got {getattr(pop, 'shape', None)}")
    pop_n, dim = int(pop.shape[0]), int(pop.shape[1])
    k_eff = max(1, min(int(k), pop_n))
    out = np.empty((n, k_eff, dim), dtype=np.float32)
    for i in range(n):
        idx = rng.choice(pop_n, size=k_eff, replace=False)
        out[i] = pop[idx]
    return out


def generate_sampled_arm_embeddings(
    *,
    episode_ids: Sequence[str],
    domains: Mapping[str, str],
    by_key: Mapping[str, Sequence[float]],
    arms: Sequence[str],
    sampler_checkpoint: str | Path,
    seed: int = 0,
    guidance_w: float = 1.5,
    ddim_steps: int | None = None,
    device: str | None = None,
    pop_max_k: int | None = None,
    project_manifold: bool | None = None,
    sample_fn: Any | None = None,
) -> dict[str, dict[str, list[float]]]:
    """Return ``{arm: {episode_id: embedding}}`` with one sample per episode.

    Episode count matches ``episode_ids`` (typically the paired/resolvable user set).
    For ``sampled_pop``, each episode is CFG-conditioned on a fresh random subset of
    size ``pop_max_k`` (default 8) drawn from that domain's real embeddings — matching
    train-time set size rather than pooling the whole population.
    """

    wanted = [a for a in arms if a in SAMPLED_ARMS]
    if not wanted:
        return {}
    if not sampler_checkpoint:
        raise ValueError(
            "sampled_uncond/sampled_pop require cue.sampler_checkpoint "
            "(trained diffusion sampler directory with sampler.pt)"
        )
    set_k = DEFAULT_POP_SET_K if pop_max_k is None else max(1, int(pop_max_k))

    by_domain: dict[str, list[str]] = defaultdict(list)
    for eid in episode_ids:
        domain = str(domains.get(eid) or "all")
        by_domain[domain].append(str(eid))

    if sample_fn is None:
        from cue_training.sampler.sample import sample as sample_fn

        own_sample_fn = True
    else:
        own_sample_fn = False

    out: dict[str, dict[str, list[float]]] = {arm: {} for arm in wanted}
    try:
        for domain, eids in sorted(by_domain.items()):
            eids = sorted(eids)
            n = len(eids)
            if n == 0:
                continue
            pop_rows = [np.asarray(by_key[eid], dtype=np.float32) for eid in eids if eid in by_key]
            if not pop_rows:
                raise ValueError(
                    f"sampled arms need real population embeddings for domain={domain!r}; "
                    f"none of {n} episodes joined"
                )
            pop = np.stack(pop_rows, axis=0)

            for arm in wanted:
                if arm == "sampled_uncond":
                    cond = None
                else:
                    rng = np.random.default_rng(_domain_seed(seed, domain, "pop_sets"))
                    cond = _per_sample_pop_sets(pop, n=n, k=set_k, rng=rng)
                result = sample_fn(
                    sampler_checkpoint,
                    n=n,
                    steps=ddim_steps,
                    guidance_w=guidance_w if cond is not None else 0.0,
                    condition_embeddings=cond,
                    device=device,
                    project_manifold=project_manifold,
                    seed=_domain_seed(seed, domain, arm),
                )
                emb = np.asarray(result["embeddings"], dtype=np.float32)
                if emb.shape[0] != n:
                    raise RuntimeError(
                        f"sampler returned {emb.shape[0]} vectors for arm={arm} domain={domain}; "
                        f"expected n={n} (one per user/episode)"
                    )
                for eid, vec in zip(eids, emb, strict=True):
                    out[arm][eid] = vec.astype(np.float32).tolist()
    finally:
        # Drop sampler weights so the decoder sidecar can claim the GPU cleanly afterward.
        if own_sample_fn:
            import gc

            gc.collect()
            try:
                import torch

                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            except Exception:  # noqa: BLE001
                pass
    return out
