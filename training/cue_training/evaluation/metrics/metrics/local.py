from __future__ import annotations

import math
import re
from collections import Counter
from pathlib import Path
from typing import Any, Mapping

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

from cue_training.baselines.ppol.features import FEATURE_NAMES, fingerprint
from cue_training.evaluation.metrics import cache
from cue_training.evaluation.metrics.data import BaselineEpisode, Episode
from cue_training.evaluation.metrics.metrics import tau_usi
from cue_training.evaluation.metrics.stats import Aggregate, aggregate

_SUCCESS_THRESHOLD = 1.0


# --------------------------------------------------------------------------- tau2

def _as_success(is_success: Any, reward: Any) -> bool | None:
    if isinstance(is_success, bool):
        return is_success
    if isinstance(reward, (int, float)):
        return float(reward) >= _SUCCESS_THRESHOLD - 1e-6
    return None


def _success_f1_from_cells(cells: Mapping[str, int]) -> float | None:
    """Binary F1 (positive = task success) from rollout-vs-human confusion counts."""

    tp = int(cells.get("both_success", 0))
    fp = int(cells.get("human_fail_rollout_success", 0))
    fn = int(cells.get("human_success_rollout_fail", 0))
    if tp + fp + fn == 0:
        return None
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    if precision + recall == 0.0:
        return 0.0
    return 2.0 * precision * recall / (precision + recall)


def tau2_task_success(
    episodes: list[Episode],
    *,
    tau_usi_source: str | None = None,
    env_error_episode_ids: set[str] | None = None,
) -> Aggregate:
    """Pairwise Success F1 vs human tau-usi rewards.

    When ``env_error_episode_ids`` is set, rollout failures tagged as Environment
    Error are treated as successes for F1 / env success (harness fault, not agent).
    The headline score is that excl-env F1; raw F1 stays in extras.
    """

    rewards = tau_usi.human_rewards(tau_usi_source)
    env_ids = {str(x) for x in (env_error_episode_ids or ()) if str(x).strip()}

    def _rollout_success(ep: Episode, raw: bool | None) -> bool | None:
        if raw is None:
            return None
        if raw:
            return True
        if ep.episode_id in env_ids:
            return True
        return False

    cells = {"both_success": 0, "human_success_rollout_fail": 0, "human_fail_rollout_success": 0, "both_fail": 0}
    cells_raw = {k: 0 for k in cells}
    vals: list[float] = []
    success_pairs: list[tuple[bool, bool]] = []
    env_success: list[float] = []
    env_success_raw: list[float] = []
    n_unpaired = 0
    n_env_rescued = 0
    for ep in episodes:
        raw = _as_success(ep.metadata.get("is_success"), ep.metadata.get("reward"))
        rollout = _rollout_success(ep, raw)
        if raw is not None:
            env_success_raw.append(1.0 if raw else 0.0)
        if rollout is not None:
            env_success.append(1.0 if rollout else 0.0)
            if raw is False and rollout is True:
                n_env_rescued += 1
        human_reward = rewards.get(ep.episode_id)
        if human_reward is None:
            human_reward = rewards.get(str(ep.metadata.get("instance_id") or ""))
        if rollout is None or not isinstance(human_reward, (int, float)):
            n_unpaired += 1
            continue
        human = float(human_reward) >= _SUCCESS_THRESHOLD - 1e-6
        success_pairs.append((human, bool(rollout)))
        vals.append(1.0 if human == bool(rollout) else 0.0)
        if human and rollout:
            cells["both_success"] += 1
        elif human:
            cells["human_success_rollout_fail"] += 1
        elif rollout:
            cells["human_fail_rollout_success"] += 1
        else:
            cells["both_fail"] += 1
        # Raw (no env rescue) confusion for extras.
        if raw is not None:
            if human and raw:
                cells_raw["both_success"] += 1
            elif human:
                cells_raw["human_success_rollout_fail"] += 1
            elif raw:
                cells_raw["human_fail_rollout_success"] += 1
            else:
                cells_raw["both_fail"] += 1
    n = len(vals)
    agreement = round(sum(vals) / n, 4) if n else None
    f1 = _success_f1_from_cells(cells)
    f1_raw = _success_f1_from_cells(cells_raw) if env_ids else f1
    confusion = {
        **cells,
        "n_paired": n,
        "agreement": agreement,
        "f1": (round(f1, 4) if f1 is not None else None),
        "n_env_rescued": n_env_rescued,
    }
    env_agg = aggregate("_env_success", env_success)
    extras = {
        "success_confusion_vs_human": confusion,
        "env_success_rate": {"mean": env_agg.mean, "ci": env_agg.confidence_interval, "n": env_agg.sample_size},
        "pairwise_success_f1": f1,
        "pairwise_success_f1_raw": f1_raw,
        "pairwise_success_agreement": agreement,
        "n_unpaired": n_unpaired,
        "n_env_error_rescued": n_env_rescued,
        "excl_env_errors": bool(env_ids),
        "tau_usi_source": tau_usi_source or "hf:cmu-lti/tau-usi",
    }
    if env_ids:
        extras["env_success_rate_raw"] = {
            "mean": aggregate("_env_success_raw", env_success_raw).mean,
            "n": len(env_success_raw),
        }
    if not rewards:
        extras["note"] = "no tau-usi human rewards resolved; set --tau-usi-source or check HF access"
    if f1 is None or not math.isfinite(f1):
        return aggregate("env/tau2_task_success", [], extras)
    rng = np.random.default_rng(0)
    bootstrap_f1 = []
    for indices in rng.integers(0, n, size=(1000, n)):
        sampled = [success_pairs[i] for i in indices]
        sampled_cells = {
            "both_success": sum(human and rollout for human, rollout in sampled),
            "human_success_rollout_fail": sum(human and not rollout for human, rollout in sampled),
            "human_fail_rollout_success": sum(not human and rollout for human, rollout in sampled),
            "both_fail": sum(not human and not rollout for human, rollout in sampled),
        }
        sampled_f1 = _success_f1_from_cells(sampled_cells)
        if sampled_f1 is not None:
            bootstrap_f1.append(sampled_f1)
    low, high = np.percentile(bootstrap_f1, [2.5, 97.5])
    return Aggregate(
        "env/tau2_task_success",
        f1,
        float(np.std(bootstrap_f1, ddof=1)),
        float(max(f1 - low, high - f1)),
        n,
        extras,
    )


def tau2_success_rate(
    episodes: list[Episode],
    *,
    tau_usi_source: str | None = None,
    env_error_episode_ids: set[str] | None = None,
) -> Aggregate:
    """Environment success rate (lifted from ``tau2_task_success`` extras)."""

    task = tau2_task_success(
        episodes,
        tau_usi_source=tau_usi_source,
        env_error_episode_ids=env_error_episode_ids,
    )
    esr = (task.extras or {}).get("env_success_rate") or {}
    mean = esr.get("mean")
    if mean is None or not math.isfinite(float(mean)):
        return aggregate("env/tau2_success_rate", [], {**(task.extras or {}), "from": "env/tau2_task_success"})
    return Aggregate(
        metric_name="env/tau2_success_rate",
        mean=float(mean),
        standard_deviation=None,
        confidence_interval=esr.get("ci"),
        sample_size=int(esr.get("n") or 0),
        extras={**(task.extras or {}), "from": "env/tau2_task_success"},
    )


# --------------------------------------------------------------------------- features

def _feature_vector(text: str) -> list[float]:
    """Sim2Real-style behavioral feature vector (regex/lexical rates)."""

    tokens = re.findall(r"\w+|[^\w\s]", text.lower())
    words = [t for t in tokens if re.search(r"\w", t)]
    n_words = max(1, len(words))
    chars = max(1, len(text))
    return [
        len(text) / 1000,
        len(words) / 200,
        sum(1 for c in text if c.isupper()) / chars,
        text.count("?") / chars,
        text.count("!") / chars,
        text.count(",") / chars,
        text.count(".") / chars,
        sum(len(w) for w in words) / n_words / 10,
        len(set(words)) / n_words,
        sum(1 for w in words if w in {"please", "thanks", "thank", "hi", "hello"}) / n_words,
        sum(1 for w in words if w in {"i", "me", "my", "we"}) / n_words,
    ]


# --------------------------------------------------------------------------- baseline state

def _human_proba(clf: Any, x: np.ndarray) -> np.ndarray:
    """P(class 1 = human), independent of sklearn's class ordering."""

    classes = list(getattr(clf, "classes_", [0, 1]))
    return clf.predict_proba(x)[:, classes.index(1) if 1 in classes else 1]


def _fit_forest_probe(x: np.ndarray, n_h: int) -> tuple[RandomForestClassifier, dict[str, Any]]:
    """PPol's discriminator: RandomForest over behavioral fingerprints (human=1 vs base=0).

    Same held-out contract as ``_fit_probe``, but a forest rather than a linear model: it
    saturates outside the human range instead of paying a simulator ever more for pushing a
    feature further than real users do (a linear logit is unbounded, so the old probe scored
    RealUserSim above real humans). Trees are scale-invariant, so no scaler is stored."""

    y = np.array([1] * n_h + [0] * (len(x) - n_h))
    strat = y if min(Counter(y).values()) > 1 else None
    train_idx, test_idx = train_test_split(np.arange(len(y)), test_size=0.2, random_state=0, stratify=strat)
    clf = RandomForestClassifier(n_estimators=200, random_state=0).fit(x[train_idx], y[train_idx])
    probs = _human_proba(clf, x[test_idx])
    refs = {
        "human_held_out_mean": float(np.mean(probs[y[test_idx] == 1])) if np.any(y[test_idx] == 1) else None,
        "base_held_out_mean": float(np.mean(probs[y[test_idx] == 0])) if np.any(y[test_idx] == 0) else None,
    }
    return clf, refs


def fit_baseline_state(baseline: dict[str, BaselineEpisode], cache_dir: Path) -> None:
    """Fit the Sim2Real discriminator probe that ``classifier/sim2real`` scores against."""

    # Human=1 vs base=0 over PPol's 19-D behavioral fingerprints. These are computed per user
    # turn rather than over one concatenated string, so turn count, per-turn length and its
    # variability are features the probe can actually see.
    eps = list(baseline.values())
    fp = np.array(
        [fingerprint(ep.human) for ep in eps] + [fingerprint(ep.base) for ep in eps], dtype=float
    )
    f_clf, f_refs = _fit_forest_probe(fp, len(eps))
    cache.save_pickle(
        cache_dir,
        "sim2real_probe_state.pkl",
        {"clf": f_clf, "refs": f_refs, "feature_names": FEATURE_NAMES},
    )


def sim2real_classifier(episodes: list[Episode], cache_dir: Path) -> Aggregate:
    """Discriminator P(human) over PPol behavioral fingerprints (probe fit at baseline)."""

    if not episodes:
        return aggregate("classifier/sim2real", [])
    state = cache.load_pickle(cache_dir, "sim2real_probe_state.pkl")
    if state is None:
        return aggregate("classifier/sim2real", [], {"error": "missing baseline state"})
    proxy = np.array([fingerprint(ep.proxy) for ep in episodes], dtype=float)
    probs = _human_proba(state["clf"], proxy)
    extras = {
        "human_held_out_mean": state["refs"].get("human_held_out_mean"),
        "held_out": {
            "human_mean": state["refs"].get("human_held_out_mean"),
            "base_mean": state["refs"].get("base_held_out_mean"),
            "proxy_mean": float(np.mean(probs)) if len(probs) else None,
            "proxy_n": int(len(probs)),
        },
        "feature_kind": "ppol_fingerprint",
    }
    return aggregate("classifier/sim2real", probs.tolist(), extras)


