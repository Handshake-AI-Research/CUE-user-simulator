from __future__ import annotations

import math
from pathlib import Path

from cue_training.evaluation.metrics.stats import Aggregate


def _fmt(value: float | int | None) -> str:
    if value is None:
        return "—"
    try:
        v = float(value)
    except (TypeError, ValueError):
        return str(value)
    if not math.isfinite(v):
        return "—"
    return f"{v:.8f}"


def _pm(mean: float | int | None, ci: float | int | None) -> str:
    """Format a value with its +/- error bar (falls back to just the value when no CI)."""

    if mean is None:
        return "—"
    m = _fmt(mean)
    if m == "—" or ci is None:
        return m
    return f"{m} ± {_fmt(ci)}"


def _scored(agg: Aggregate | None) -> bool:
    return bool(agg) and not agg.extras.get("skipped") and not agg.extras.get("error")


def _cell(agg: Aggregate | None) -> str:
    return _pm(agg.mean, agg.confidence_interval) if _scored(agg) else "—"


def _group_title(arm: str, domain: str, n_episodes: int) -> str:
    if domain and domain != "all":
        return f"# Arm: {arm} · Domain: {domain} ({n_episodes} episodes)"
    return f"# Arm: {arm} ({n_episodes} episodes)"


_DISPLAY_ROWS = (
    ("env/tau2_success_rate", "Success rate"),
    ("env/tau2_task_success", "Success F1"),
    ("classifier/sim2real", "Sim2Real classifier P(human)"),
    ("judge/turing_sonnet_qwen", "Naturalness Turing |0.5-mean_P(human)|"),
    ("mimicry/wegmann_ava", "Wegmann AVA"),
    ("mimicry/paired_audit", "Paired audit fidelity"),
    ("coverage/sim2real_behavioral", "Sim2Real / PPol Chamfer coverage"),
    ("coverage/styledistance_behavioral", "StyleDistance + PCA coverage"),
)


def render_markdown(
    *,
    run_id: str,
    baseline_path: Path,
    candidate_path: Path | None,
    aggregates_by_group: dict[str, list[Aggregate]],
    n_by_group: dict[str, int],
    group_meta: dict[str, dict[str, str]],
) -> str:
    groups = list(aggregates_by_group)
    arms = sorted({group_meta[g]["arm"] for g in groups})
    domains = sorted({group_meta[g]["domain"] for g in groups if group_meta[g]["domain"] != "all"})
    multi_arm = len(arms) > 1 or (len(arms) == 1 and arms[0] != "all")
    multi_domain = len(domains) > 1
    show_header = multi_arm or multi_domain or (len(groups) == 1 and groups[0] != "all")
    lines = ["# Metrics Summary"]
    for group, aggregates in aggregates_by_group.items():
        meta = group_meta.get(group, {"arm": group, "domain": "all"})
        by_name = {agg.metric_name: agg for agg in aggregates}
        # Back-compat: success rate may live only in task_success extras.
        if "env/tau2_success_rate" not in by_name and "env/tau2_task_success" in by_name:
            env = by_name["env/tau2_task_success"]
            esr = (env.extras or {}).get("env_success_rate") or {}
            if esr.get("mean") is not None:
                by_name["env/tau2_success_rate"] = Aggregate(
                    metric_name="env/tau2_success_rate",
                    mean=float(esr["mean"]),
                    standard_deviation=None,
                    confidence_interval=esr.get("ci"),
                    sample_size=int(esr.get("n") or 0),
                    extras={},
                )
        if show_header:
            lines += ["", _group_title(
                meta["arm"], meta.get("domain", "all"), n_by_group.get(group, 0)
            )]
        lines += [
            "",
            "| Metric | Value ± 95% CI |",
            "| --- | --- |",
        ]
        for key, label in _DISPLAY_ROWS:
            if key in by_name:
                lines.append(f"| {label} | {_cell(by_name.get(key))} |")
    lines.append("")
    return "\n".join(lines)
