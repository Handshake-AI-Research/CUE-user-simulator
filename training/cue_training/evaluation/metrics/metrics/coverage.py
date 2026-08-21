"""Coverage: bidirectional Chamfer overlap between proxy and human trajectory clouds."""

from __future__ import annotations

import numpy as np

from cue_training.evaluation.metrics.data import Episode, user_text
from cue_training.evaluation.metrics.metrics.style_encoder import (
    DEFAULT_STYLEDISTANCE_MODEL,
    get_styledistance_encoder,
)
from cue_training.evaluation.metrics.stats import Aggregate, aggregate
from cue_training.runlog.log import warn

DEFAULT_BEHAVIORAL_PCA_DIM = 16


def _group_key(ep: Episode, *, split_domain: bool) -> str:
    if split_domain and ep.domain and ep.domain != "all":
        return f"{ep.arm}/{ep.domain}"
    return ep.arm


def _domain_key(ep: Episode, *, split_domain: bool) -> str:
    if split_domain and ep.domain and ep.domain != "all":
        return str(ep.domain)
    return "all"


def _l2_distances(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    squared = (
        np.sum(a * a, axis=1, keepdims=True)
        + np.sum(b * b, axis=1)[None, :]
        - 2.0 * (a @ b.T)
    )
    return np.sqrt(np.maximum(squared, 0.0))


def _mean_pairwise_distance(x: np.ndarray, *, block_size: int = 512) -> float:
    """Mean Euclidean distance over unique pairs without materializing the full matrix."""

    total = 0.0
    count = 0
    for i in range(0, len(x), block_size):
        a = x[i : i + block_size]
        for j in range(i, len(x), block_size):
            b = x[j : j + block_size]
            distances = _l2_distances(a, b)
            if i == j:
                distances = distances[np.triu_indices(len(a), k=1)]
            total += float(distances.sum())
            count += int(distances.size)
    return total / count if count else 0.0


def _encode_styledistance_concat(
    episodes: list[Episode],
    *,
    model_name: str,
    side: str,
) -> np.ndarray:
    if side == "human":
        texts = [user_text(ep.human) for ep in episodes]
    elif side == "proxy":
        texts = [user_text(ep.proxy) for ep in episodes]
    else:
        raise ValueError(f"side must be 'human' or 'proxy', got {side!r}")
    return np.asarray(get_styledistance_encoder(model_name).encode(texts), dtype=np.float32)


def _human_refs_concat(episodes: list[Episode], *, split_domain: bool) -> dict[str, list[Episode]]:
    out: dict[str, list[Episode]] = {}
    seen: dict[str, set[str]] = {}
    for ep in episodes:
        domain = _domain_key(ep, split_domain=split_domain)
        sid = seen.setdefault(domain, set())
        if ep.episode_id in sid:
            continue
        if not user_text(ep.human):
            continue
        sid.add(ep.episode_id)
        out.setdefault(domain, []).append(ep)
    return out


def _pca_fit_humans(human_x: np.ndarray, *, pca_dim: int):
    from sklearn.decomposition import PCA

    k = max(1, min(int(pca_dim), len(human_x) - 1, human_x.shape[-1]))
    pca = PCA(n_components=k, random_state=0)
    human_z = pca.fit_transform(human_x)
    return pca, human_z, k


def _skipped_coverage(
    episodes: list[Episode],
    *,
    metric_name: str,
    reason: str,
    embedding: str,
) -> dict[str, Aggregate]:
    domains = {ep.domain for ep in episodes if ep.domain and ep.domain != "all"}
    split_domain = len(domains) > 1
    keys = {_group_key(ep, split_domain=split_domain) for ep in episodes}
    return {
        key: aggregate(metric_name, [], {"skipped": reason, "embedding": embedding})
        for key in keys
    }


def _coverage_behavioral(
    episodes: list[Episode],
    *,
    metric_name: str,
    embedding: str,
    model_name: str,
    encode,
    pca_dim: int | None = DEFAULT_BEHAVIORAL_PCA_DIM,
) -> dict[str, Aggregate]:
    """Bidirectional Chamfer coverage; optional human-fitted PCA (None = raw space)."""

    domains = {ep.domain for ep in episodes if ep.domain and ep.domain != "all"}
    split_domain = len(domains) > 1
    by_group: dict[str, list[Episode]] = {}
    for ep in episodes:
        by_group.setdefault(_group_key(ep, split_domain=split_domain), []).append(ep)
    human_by_domain = _human_refs_concat(episodes, split_domain=split_domain)

    out: dict[str, Aggregate] = {}
    for key, eps in by_group.items():
        arm = key.split("/")[0]
        domain = key.split("/")[1] if "/" in key else "all"
        ref = human_by_domain.get(domain) or human_by_domain.get("all") or []
        try:
            proxy_x = encode(eps, "proxy")
            human_x = encode(ref, "human")
            proxy_mask = np.linalg.norm(proxy_x, axis=1) > 0
            human_mask = np.linalg.norm(human_x, axis=1) > 0
            human_x = human_x[human_mask]
            proxy_x = proxy_x[proxy_mask]
            if len(human_x) < 2 or not len(proxy_x):
                out[key] = aggregate(
                    metric_name,
                    [],
                    {
                        "skipped": "need at least one proxy and two non-empty human trajectories",
                        "n_proxy": int(len(proxy_x)),
                        "n_human": int(len(human_x)),
                        "arm": arm,
                    },
                )
                continue

            pca_k = None
            explained = None
            if pca_dim is not None and int(pca_dim) > 0:
                pca, human_z, pca_k = _pca_fit_humans(human_x, pca_dim=int(pca_dim))
                proxy_z = pca.transform(proxy_x)
                explained = float(pca.explained_variance_ratio_.sum())
            else:
                human_z = np.asarray(human_x, dtype=np.float32)
                proxy_z = np.asarray(proxy_x, dtype=np.float32)

            d_ref = _mean_pairwise_distance(human_z)
            if d_ref <= 0:
                out[key] = aggregate(
                    metric_name,
                    [],
                    {"skipped": "human reference spread is zero", "n_human": int(len(human_z)), "arm": arm},
                )
                continue

            distances = _l2_distances(human_z, proxy_z)
            error = float(distances.min(axis=1).mean() + distances.min(axis=0).mean())
            score = max(0.0, 1.0 - min(1.0, error / (2.0 * d_ref)))

            out[key] = aggregate(
                metric_name,
                [score],
                {
                    "arm": arm,
                    "domain": domain,
                    "embedding": embedding,
                    "reference": "rollout_humans",
                    "model": model_name,
                    "pca": bool(pca_k),
                    "pca_dim": pca_k,
                    "pca_fit": "human_reference_only" if pca_k else None,
                    "explained_variance_ratio_sum": explained,
                    "d_ref": d_ref,
                    "n_proxy": int(len(proxy_z)),
                    "n_human": int(len(human_z)),
                    "aggregation": "domain_level_chamfer",
                },
            )
        except Exception as exc:  # noqa: BLE001
            warn("metrics", f"{metric_name} failed for {key}: {exc}")
            out[key] = aggregate(
                metric_name,
                [],
                {"error": str(exc), "embedding": embedding, "arm": arm},
            )
    return out


def coverage_styledistance_behavioral(
    episodes: list[Episode],
    *,
    model_name: str = DEFAULT_STYLEDISTANCE_MODEL,
    pca_dim: int = DEFAULT_BEHAVIORAL_PCA_DIM,
) -> dict[str, Aggregate]:
    """PPol behavioral coverage in StyleDistance space with human-fitted PCA."""

    def _encode(eps: list[Episode], side: str) -> np.ndarray:
        return _encode_styledistance_concat(eps, model_name=model_name, side=side)

    try:
        get_styledistance_encoder(model_name)
    except Exception:  # noqa: BLE001
        return _skipped_coverage(
            episodes,
            metric_name="coverage/styledistance_behavioral",
            reason=f"styledistance unavailable ({model_name})",
            embedding="styledistance",
        )
    return _coverage_behavioral(
        episodes,
        metric_name="coverage/styledistance_behavioral",
        embedding="styledistance_concat_user_turns",
        model_name=model_name,
        encode=_encode,
        pca_dim=pca_dim,
    )


def coverage_sim2real_behavioral(
    episodes: list[Episode],
) -> dict[str, Aggregate]:
    """PPol fingerprint Chamfer coverage (original 19-D regex space, no PCA).

    Matches the PPol / persona-policies bidirectional Chamfer reward: fingerprints
    from ``cue_training.baselines.ppol.features.fingerprint``, score
    ``1 - min(1, err / (2 * d_ref))`` with ``d_ref`` = mean human pairwise L2.
    """

    from cue_training.baselines.ppol.features import fingerprint

    def _encode(eps: list[Episode], side: str) -> np.ndarray:
        trajs = [ep.human if side == "human" else ep.proxy for ep in eps]
        return np.asarray([fingerprint(t) for t in trajs], dtype=np.float32)

    return _coverage_behavioral(
        episodes,
        metric_name="coverage/sim2real_behavioral",
        embedding="ppol_fingerprint",
        model_name="ppol_fingerprint",
        encode=_encode,
        pca_dim=None,
    )
