"""Leakage metrics for full-session versus causal encoder conditioning."""

from __future__ import annotations

import json
import random
import re
from pathlib import Path
from typing import Any

import numpy as np

_TOKEN_RE = re.compile(r"\b[\w'-]+\b")


def paired_gap(
    full: list[float],
    control: list[float],
    *,
    bootstrap_samples: int = 1_000,
    seed: int = 0,
) -> dict[str, float]:
    """Mean paired gap and percentile bootstrap interval."""

    if len(full) != len(control) or not full:
        return {"mean": float("nan"), "ci_low": float("nan"), "ci_high": float("nan")}
    diff = np.asarray(full, dtype=np.float64) - np.asarray(control, dtype=np.float64)
    rng = np.random.default_rng(seed)
    means = np.empty(max(1, bootstrap_samples), dtype=np.float64)
    for i in range(len(means)):
        means[i] = diff[rng.integers(0, len(diff), size=len(diff))].mean()
    return {
        "mean": float(diff.mean()),
        "ci_low": float(np.quantile(means, 0.025)),
        "ci_high": float(np.quantile(means, 0.975)),
    }


def target_manual_overlap(manual_text: str, target: str, *, n: int = 4) -> dict[str, float]:
    """Normalized n-gram overlap plus direct target-substring leakage."""

    manual_tokens = _TOKEN_RE.findall(manual_text.lower())
    target_tokens = _TOKEN_RE.findall(target.lower())
    if not target_tokens:
        return {"ngram_recall": 0.0, "target_substring": 0.0}
    width = min(max(1, n), len(target_tokens))
    target_ngrams = {
        tuple(target_tokens[i : i + width])
        for i in range(len(target_tokens) - width + 1)
    }
    manual_ngrams = {
        tuple(manual_tokens[i : i + width])
        for i in range(max(0, len(manual_tokens) - width + 1))
    }
    recall = len(target_ngrams & manual_ngrams) / max(1, len(target_ngrams))
    normalized_target = " ".join(target_tokens)
    normalized_manual = " ".join(manual_tokens)
    return {
        "ngram_recall": float(recall),
        "target_substring": float(
            len(target_tokens) >= 3 and normalized_target in normalized_manual
        ),
    }


def role_compatible_future(
    prefix: list[dict[str, Any]],
    donor_turns: list[dict[str, Any]],
    *,
    seed: int,
) -> list[dict[str, Any]]:
    """Attach a deterministic donor suffix while preserving the prefix verbatim."""

    rng = random.Random(seed)
    candidates = [i for i, turn in enumerate(donor_turns) if turn.get("role") == "user"]
    start = rng.choice(candidates) if candidates else 0
    return [dict(turn) for turn in prefix] + [dict(turn) for turn in donor_turns[start:]]


def mask_target(
    turns: list[dict[str, Any]],
    target_index: int,
    *,
    marker: str = "<MASKED_USER_TURN>",
) -> list[dict[str, Any]]:
    out = [dict(turn) for turn in turns]
    if 0 <= target_index < len(out):
        out[target_index]["content"] = marker
    return out


def save_report(output_dir: str | Path, epoch: int, report: dict[str, Any]) -> Path:
    path = Path(output_dir) / "leakage_audit" / f"epoch_{epoch:04d}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    return path
