from __future__ import annotations

import math
from dataclasses import dataclass, field
from statistics import mean, stdev
from typing import Any


@dataclass
class Aggregate:
    metric_name: str
    mean: float
    standard_deviation: float | None = None
    confidence_interval: float | None = None
    sample_size: int = 0
    extras: dict[str, Any] = field(default_factory=dict)

    def to_json(self) -> dict[str, Any]:
        return {
            "metric_name": self.metric_name,
            "mean": self.mean,
            "standard_deviation": self.standard_deviation,
            "confidence_interval": self.confidence_interval,
            "sample_size": self.sample_size,
            "extras": self.extras,
        }


def aggregate(metric_name: str, values: list[float], extras: dict[str, Any] | None = None) -> Aggregate:
    clean = [float(v) for v in values if v is not None and math.isfinite(float(v))]
    n = len(clean)
    if n == 0:
        return Aggregate(metric_name=metric_name, mean=float("nan"), sample_size=0, extras=extras or {})
    sd = stdev(clean) if n > 1 else None
    ci = 1.96 * sd / math.sqrt(n) if sd is not None else None
    return Aggregate(metric_name=metric_name, mean=mean(clean), standard_deviation=sd, confidence_interval=ci, sample_size=n, extras=extras or {})


def aggregate_seed_means(
    metric_name: str,
    values: list[float],
    extras: dict[str, Any] | None = None,
) -> Aggregate:
    """Mean of seed-level scores with ± = sample stdev of those means (not SEM CI).

    Used by the aggregate leaderboard when pooling across seeds (and for domain macros,
    after averaging domains within each seed).
    """

    clean = [float(v) for v in values if v is not None and math.isfinite(float(v))]
    n = len(clean)
    if n == 0:
        return Aggregate(metric_name=metric_name, mean=float("nan"), sample_size=0, extras=extras or {})
    sd = stdev(clean) if n > 1 else None
    return Aggregate(
        metric_name=metric_name,
        mean=mean(clean),
        standard_deviation=sd,
        # Leaderboard cells render ``confidence_interval`` as the ± bar.
        confidence_interval=sd,
        sample_size=n,
        extras=extras or {},
    )


def aggregate_from_json(payload: dict[str, Any]) -> Aggregate:
    return Aggregate(
        metric_name=str(payload.get("metric_name") or ""),
        mean=float(payload["mean"]) if payload.get("mean") is not None else float("nan"),
        standard_deviation=payload.get("standard_deviation"),
        confidence_interval=payload.get("confidence_interval"),
        sample_size=int(payload.get("sample_size") or 0),
        extras=dict(payload.get("extras") or {}),
    )

