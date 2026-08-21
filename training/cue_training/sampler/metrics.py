"""Dev-set metrics and acceptance thresholds for the cue embedding sampler."""

from __future__ import annotations

from typing import Any

import numpy as np
import torch
import torch.nn.functional as F

from cue_training.data.schema import parse_manual
from cue_training.sampler.bank import EmbeddingBank, _rff_mmd, _sliced_wasserstein
from cue_training.sampler.faiss_index import EmbeddingIndex


def sliced_wasserstein(a: np.ndarray, b: np.ndarray, **kwargs: Any) -> float:
    return _sliced_wasserstein(a, b, **kwargs)


def rff_mmd(a: np.ndarray, b: np.ndarray, **kwargs: Any) -> float:
    return _rff_mmd(a, b, **kwargs)


def knn_coverage(
    samples: np.ndarray,
    reference: np.ndarray,
    *,
    k: int = 5,
) -> dict[str, float]:
    """Simple k-NN precision/recall proxies in cosine space."""

    if samples.shape[0] == 0 or reference.shape[0] == 0:
        return {"precision": float("nan"), "recall": float("nan"), "coverage": float("nan")}
    ref_index = EmbeddingIndex(reference.shape[1])
    ref_index.add(reference)
    samp_index = EmbeddingIndex(samples.shape[1])
    samp_index.add(samples)
    # precision: fraction of samples whose NN in ref is "close" (top-1 cos > median ref self-NN)
    sims_s, _ = ref_index.search(samples, 1)
    sims_r, _ = ref_index.search(reference[: min(2048, len(reference))], 2)
    thresh = float(np.median(sims_r[:, 1])) if sims_r.shape[1] > 1 else 0.0
    precision = float((sims_s[:, 0] >= thresh).mean())
    # recall/coverage: fraction of ref points that have a sample among their k NNs in joint set
    # cheaper proxy: mean max cosine from ref->samples
    sims_rs, _ = samp_index.search(reference[: min(4096, len(reference))], 1)
    recall = float((sims_rs[:, 0] >= thresh).mean())
    coverage = float(sims_rs[:, 0].mean())
    return {"precision": precision, "recall": recall, "coverage": coverage, "nn_thresh": thresh}


def nearest_distances(
    samples: np.ndarray,
    train: np.ndarray,
    dev: np.ndarray,
) -> dict[str, float]:
    if samples.shape[0] == 0:
        return {
            "median_nn_train": float("nan"),
            "median_nn_dev": float("nan"),
            "memorization_ratio": float("nan"),
        }
    train_index = EmbeddingIndex(train.shape[1])
    train_index.add(train)
    dev_index = EmbeddingIndex(dev.shape[1])
    if dev.shape[0]:
        dev_index.add(dev)
    sims_t, _ = train_index.search(samples, 1)
    # convert cosine sim -> distance
    d_train = 1.0 - sims_t[:, 0]
    if dev.shape[0]:
        sims_d, _ = dev_index.search(samples, 1)
        d_dev = 1.0 - sims_d[:, 0]
    else:
        d_dev = d_train
    med_t = float(np.median(d_train))
    med_d = float(np.median(d_dev))
    ratio = med_t / max(med_d, 1e-8)
    return {
        "median_nn_train": med_t,
        "median_nn_dev": med_d,
        "memorization_ratio": ratio,
    }


def conditional_fidelity(
    samples: np.ndarray,
    condition_sets: list[np.ndarray],
    global_mean: np.ndarray,
) -> dict[str, float]:
    """sample->set distance vs sample->global-mean; plus pairwise diversity."""

    if not samples.shape[0] or not condition_sets:
        return {
            "mean_set_distance": float("nan"),
            "mean_global_distance": float("nan"),
            "bias_gap": float("nan"),
            "diversity": float("nan"),
        }
    set_dists = []
    glob_dists = []
    for i, sample in enumerate(samples):
        cond = condition_sets[min(i, len(condition_sets) - 1)]
        if cond.size == 0:
            continue
        cmean = cond.mean(axis=0)
        set_dists.append(float(1.0 - _cos(sample, cmean)))
        glob_dists.append(float(1.0 - _cos(sample, global_mean)))
    # diversity: mean pairwise cosine distance among samples
    if samples.shape[0] > 1:
        norms = samples / (np.linalg.norm(samples, axis=1, keepdims=True) + 1e-8)
        sims = norms @ norms.T
        iu = np.triu_indices(samples.shape[0], k=1)
        diversity = float(1.0 - sims[iu].mean())
    else:
        diversity = 0.0
    mean_set = float(np.mean(set_dists)) if set_dists else float("nan")
    mean_glob = float(np.mean(glob_dists)) if glob_dists else float("nan")
    return {
        "mean_set_distance": mean_set,
        "mean_global_distance": mean_glob,
        "bias_gap": mean_glob - mean_set,
        "diversity": diversity,
    }


def _cos(a: np.ndarray, b: np.ndarray) -> float:
    return float(
        np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-8)
    )


@torch.no_grad()
def decode_parse_rate(
    model: Any,
    embeddings: torch.Tensor | np.ndarray,
    *,
    max_new_tokens: int = 512,
    batch_size: int = 4,
) -> dict[str, Any]:
    """Decode bottlenecks and measure parse_manual success rate."""

    if isinstance(embeddings, np.ndarray):
        emb = torch.from_numpy(np.asarray(embeddings, dtype=np.float32))
    else:
        emb = embeddings.detach().float()
    device = next(model.parameters()).device
    emb = emb.to(device)
    from cue_training.data.schema import COMMAND_BLOCK_SENTINEL

    tok = model.decoder.tokenizer
    sentinel = tok(COMMAND_BLOCK_SENTINEL, add_special_tokens=False, return_tensors="pt")[
        "input_ids"
    ][0].to(device)
    parsed = 0
    total = emb.shape[0]
    lengths: list[int] = []
    texts: list[str] = []
    for start in range(0, total, batch_size):
        batch = emb[start : start + batch_size]
        outs = model.decoder.generate(
            bottleneck=batch, sentinel_ids=sentinel, max_new_tokens=max_new_tokens
        )
        for text in outs:
            texts.append(text)
            lengths.append(len(text))
            if parse_manual(text) is not None:
                parsed += 1
    dup_rate = 0.0
    if texts:
        dup_rate = 1.0 - (len(set(texts)) / len(texts))
    return {
        "parse_rate": parsed / max(total, 1),
        "mean_length": float(np.mean(lengths)) if lengths else 0.0,
        "duplicate_rate": float(dup_rate),
        "n": float(total),
        # Kept so a sharded caller can pool texts and get the true global duplicate rate;
        # per-shard rates miss duplicates that straddle shards.
        "texts": texts,
    }


def denoising_mse(
    model: torch.nn.Module,
    schedule: Any,
    z0: torch.Tensor,
    *,
    set_emb: torch.Tensor | None = None,
    mask: torch.Tensor | None = None,
) -> float:
    device = z0.device
    t = torch.randint(0, schedule.timesteps, (z0.shape[0],), device=device)
    zt, noise = schedule.q_sample(z0, t)
    pred = model(zt, t, set_emb=set_emb, mask=mask)
    return float(F.mse_loss(pred, noise).detach())


def acceptance_report(
    metrics: dict[str, Any],
    thresholds: dict[str, Any],
    baselines: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Pass/fail against config-declared thresholds (relative to bank baselines)."""

    baselines = baselines or {}
    checks: dict[str, Any] = {}

    # Decodability
    parse = float(metrics.get("parse_rate_joint_training", metrics.get("parse_rate", 0.0)))
    parse_base = baselines.get("real_embedding_parse_rate")
    min_parse = thresholds.get("min_parse_rate")
    if min_parse is None and parse_base is not None:
        margin = float(thresholds.get("parse_rate_margin", 0.1))
        min_parse = max(0.0, float(parse_base) - margin)
    if min_parse is not None:
        checks["parse_rate_joint_training"] = {
            "value": parse,
            "threshold": float(min_parse),
            "pass": parse >= float(min_parse),
        }
    parse2 = metrics.get("parse_rate_refinement")
    if parse2 is not None and min_parse is not None:
        checks["parse_rate_refinement"] = {
            "value": float(parse2),
            "threshold": float(min_parse),
            "pass": float(parse2) >= float(min_parse),
        }

    # Distributional realism vs train-vs-dev baseline
    sw = metrics.get("sliced_wasserstein")
    sw_base = baselines.get("real_train_vs_dev_sliced_wasserstein")
    sw_mult = float(thresholds.get("sw_vs_baseline_max_mult", 2.0))
    if sw is not None and sw_base is not None and np.isfinite(sw_base) and sw_base > 0:
        checks["sliced_wasserstein"] = {
            "value": float(sw),
            "threshold": float(sw_base) * sw_mult,
            "pass": float(sw) <= float(sw_base) * sw_mult,
        }
    mmd = metrics.get("rff_mmd")
    mmd_base = baselines.get("real_train_vs_dev_rff_mmd")
    mmd_mult = float(thresholds.get("mmd_vs_baseline_max_mult", 2.0))
    if mmd is not None and mmd_base is not None and np.isfinite(mmd_base) and mmd_base > 0:
        checks["rff_mmd"] = {
            "value": float(mmd),
            "threshold": float(mmd_base) * mmd_mult,
            "pass": float(mmd) <= float(mmd_base) * mmd_mult,
        }

    # Memorization: median nn-train should not be << median nn-dev
    mem_ratio = metrics.get("memorization_ratio")
    min_mem = float(thresholds.get("min_memorization_ratio", 0.5))
    if mem_ratio is not None and np.isfinite(mem_ratio):
        checks["memorization_ratio"] = {
            "value": float(mem_ratio),
            "threshold": min_mem,
            "pass": float(mem_ratio) >= min_mem,
        }

    # Conditional bias
    bias_gap = metrics.get("bias_gap")
    min_gap = float(thresholds.get("min_bias_gap", 0.02))
    if bias_gap is not None and np.isfinite(bias_gap):
        checks["bias_gap"] = {
            "value": float(bias_gap),
            "threshold": min_gap,
            "pass": float(bias_gap) >= min_gap,
        }
    diversity = metrics.get("diversity")
    min_div = float(thresholds.get("min_conditional_diversity", 0.05))
    if diversity is not None and np.isfinite(diversity):
        checks["diversity"] = {
            "value": float(diversity),
            "threshold": min_div,
            "pass": float(diversity) >= min_div,
        }

    passed = all(c["pass"] for c in checks.values()) if checks else False
    return {"pass": passed, "checks": checks, "metrics": metrics, "baselines": baselines}


def evaluate_samples_vs_bank(
    samples: np.ndarray,
    bank: EmbeddingBank,
    *,
    condition_sets: list[np.ndarray] | None = None,
) -> dict[str, Any]:
    train = np.asarray(bank.prior_train.embeddings, dtype=np.float32)
    dev = np.asarray(bank.prior_dev.embeddings, dtype=np.float32)
    if dev.shape[0] == 0:
        dev = train[: min(1024, train.shape[0])]
    mu, sigma = bank.mu, bank.sigma
    s_s = (samples - mu) / sigma
    d_s = (dev - mu) / sigma
    t_s = (train - mu) / sigma
    rng = np.random.default_rng(0)

    def sub(x: np.ndarray, n: int = 2048) -> np.ndarray:
        if x.shape[0] <= n:
            return x
        return x[rng.choice(x.shape[0], size=n, replace=False)]

    out: dict[str, Any] = {
        "sliced_wasserstein": sliced_wasserstein(sub(s_s), sub(d_s)),
        "rff_mmd": rff_mmd(sub(s_s), sub(d_s)),
    }
    out.update(knn_coverage(sub(s_s, 1024), sub(d_s, 4096)))
    out.update(nearest_distances(sub(s_s, 1024), sub(t_s, 8192), sub(d_s, 4096)))
    if condition_sets is not None:
        out.update(
            conditional_fidelity(
                samples, condition_sets, global_mean=train.mean(axis=0)
            )
        )
    return out
