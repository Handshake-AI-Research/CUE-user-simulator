"""Compositional pairwise-success bars for τ² (human vs sim, per episode).

Two stack segments (outer ring of the bar = 100% of paired episodes):
  correctly    both succeed or both fail
  incorrectly  one succeeds and the other fails

One panel per base simulator. Method set matches the FMA bar figure.

    uv run --group fma cue-fma pairwise-success
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np

from cue_training.evaluation.fma.plots.plot_failure_types import (
    BASE_MODELS,
    DPI,
    FONT_FAMILY,
    MODEL_NAMES,
    PAIRED_SPECS,
    _pyplot,
    is_included,
    label_of,
    parse_source,
    series_for,
)

STEM = "pairwise_success_composition"
CELL_KEYS = (
    "both_fail",
    "both_success",
    "human_success_rollout_fail",
    "human_fail_rollout_success",
)
PLOT_KEYS = ("agree", "disagree")
SEGMENT_LABELS = {
    "agree": "Correctly",
    "disagree": "Incorrectly",
}
SEGMENT_COLORS = {
    "agree": "#d3fb52",
    "disagree": "#ff7d4d",
}
_EDGE = "#052326"
TINY = 7


def _confusion_from_aggs(aggs: list[Any]) -> dict[str, int] | None:
    for item in aggs or []:
        if not isinstance(item, dict):
            continue
        if item.get("metric_name") != "env/tau2_task_success":
            continue
        conf = (item.get("extras") or {}).get("success_confusion_vs_human") or {}
        if not any(k in conf for k in CELL_KEYS):
            continue
        return {k: int(conf.get(k) or 0) for k in CELL_KEYS}
    return None


def load_confusion(runs_root: Path) -> dict[str, Counter]:
    """Sum confusion cells across seeds and τ² domains for each FMA source_id."""

    out: dict[str, Counter] = {}
    for path in sorted(runs_root.glob("**/summary.json")):
        parts = path.parts
        try:
            sim = parts[-2]
            family = parts[-3]
            seed_dir = parts[-4]
        except IndexError:
            continue
        if not seed_dir.startswith("seed-"):
            continue
        if sim not in BASE_MODELS:
            continue
        payload = json.loads(path.read_text(encoding="utf-8"))
        by_group = (
            payload.get("aggregates_by_group") or payload.get("aggregates_by_arm") or {}
        )
        group_meta = payload.get("group_meta") or {}
        for group, aggs in by_group.items():
            conf = _confusion_from_aggs(aggs)
            if conf is None:
                continue
            meta = group_meta.get(group) or {}
            arm = str(meta.get("arm") or str(group).split("/")[0])
            sid = f"{family}-{sim}|{arm}"
            if not is_included(sid):
                continue
            out.setdefault(sid, Counter())
            for key in CELL_KEYS:
                out[sid][key] += int(conf[key])
    return out


def keep_series(
    sid: str,
    *,
    simulators: tuple[str, ...] = ("gemini",),
    paired_only: bool = True,
) -> bool:
    family, sim, arm = parse_source(sid)
    if sim not in simulators:
        return False
    if paired_only:
        return (family, arm) in PAIRED_SPECS
    return True


def as_percent(counts: dict[str, Counter]) -> dict[str, dict[str, float]]:
    shares: dict[str, dict[str, float]] = {}
    for sid, cells in counts.items():
        total = sum(int(cells.get(k, 0)) for k in CELL_KEYS)
        if total <= 0:
            continue
        agree = int(cells.get("both_fail", 0)) + int(cells.get("both_success", 0))
        disagree = int(cells.get("human_success_rollout_fail", 0)) + int(
            cells.get("human_fail_rollout_success", 0)
        )
        shares[sid] = {
            "agree": 100.0 * agree / total,
            "disagree": 100.0 * disagree / total,
        }
    return shares


def plot_composition(
    percents: dict[str, dict[str, float]],
    out_dir: Path,
    *,
    stem: str = STEM,
    simulators: tuple[str, ...] = ("gemini",),
) -> list[Path]:
    plt = _pyplot()
    from matplotlib.patches import Patch

    bases = [b for b in simulators if b in BASE_MODELS and series_for(percents, b)]
    if not bases:
        raise SystemExit("no pairwise-success series to plot")
    n_rows = len(bases)
    fig, axes = plt.subplots(
        n_rows,
        1,
        figsize=(4.3, 1.9 * n_rows),
        sharey=True,
    )
    axes = np.atleast_1d(axes)

    for ax, base in zip(axes, bases):
        series = series_for(percents, base)
        x = np.arange(len(series))
        bottoms = np.zeros(len(series))
        ax.grid(True, axis="y", color="lightgray", linestyle=":", linewidth=0.5)
        ax.set_axisbelow(True)
        for key in PLOT_KEYS:
            heights = np.array(
                [percents[sid].get(key, 0.0) for sid in series], dtype=float
            )
            ax.bar(
                x,
                heights,
                0.5,
                bottom=bottoms,
                color=SEGMENT_COLORS[key],
                edgecolor=_EDGE,
                linewidth=0.4,
            )
            bottoms += heights
        ax.set_xlim(-0.6, len(series) - 0.4)
        ax.set_ylim(0, 100)
        ax.set_yticks([0, 25, 50, 75, 100])
        ax.set_ylabel("% of Episodes", fontsize=TINY)
        if n_rows > 1:
            ax.set_title(
                MODEL_NAMES[base], fontsize=TINY, fontweight=600, loc="left", pad=3
            )
        ax.tick_params(labelsize=TINY, pad=2)
        ax.tick_params(axis="y", length=0)
        ax.tick_params(axis="x", which="both", bottom=False, labelbottom=False)
        for side in ("top", "right", "left"):
            ax.spines[side].set_visible(False)
        ax.spines["bottom"].set_color("#bbbbbb")
        ax.set_xticks(x)

    labels = [label_of(sid) for sid in series_for(percents, bases[-1])]
    axes[-1].tick_params(axis="x", which="both", bottom=False, labelbottom=True)
    axes[-1].set_xticklabels(labels, fontsize=TINY)

    axes[0].legend(
        [
            Patch(facecolor=SEGMENT_COLORS[k], edgecolor=_EDGE, linewidth=0.4)
            for k in PLOT_KEYS
        ],
        [SEGMENT_LABELS[k] for k in PLOT_KEYS],
        loc="lower center",
        bbox_to_anchor=(0.5, 1.0),
        ncol=2,
        fontsize=TINY,
        frameon=False,
        columnspacing=1.0,
        handlelength=1.1,
        handletextpad=0.4,
        borderpad=0.0,
        title="Predicts User's Task Success",
        title_fontsize=TINY,
        alignment="center",
    )
    if n_rows > 1:
        fig.suptitle(
            "τ²-Bench", fontsize=9, fontfamily=FONT_FAMILY, fontweight=600, y=1.06
        )
    else:
        axes[0].set_title(
            "τ²-Bench",
            fontsize=9,
            fontfamily=FONT_FAMILY,
            fontweight=600,
            pad=26,
        )
    fig.subplots_adjust(left=0.13, right=0.99, top=0.78, bottom=0.12, hspace=0.34)
    out_dir.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []
    for ext in ("png", "pdf"):
        path = out_dir / f"{stem}.{ext}"
        fig.savefig(path, dpi=DPI, bbox_inches="tight", pad_inches=0.05)
        written.append(path)
    plt.close(fig)
    return written


def run_pairwise_success(
    *,
    runs_root: Path,
    out_dir: Path,
    stem: str = STEM,
) -> dict[str, Any]:
    counts = load_confusion(Path(runs_root))
    counts = {sid: c for sid, c in counts.items() if keep_series(sid)}
    if not counts:
        raise ValueError(f"no confusion cells under {runs_root}")
    percents = as_percent(counts)
    written = plot_composition(percents, Path(out_dir), stem=stem)
    return {
        "n_series": len(counts),
        "n_episodes": {sid: int(sum(c.values())) for sid, c in counts.items()},
        "figures": [str(p) for p in written],
    }


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--runs-root",
        type=Path,
        default=Path("output/metrics/tau2_customer-service/_runs"),
    )
    parser.add_argument("--out-dir", type=Path, default=Path("fma/figures"))
    parser.add_argument("--stem", default=STEM)
    args = parser.parse_args(argv)
    result = run_pairwise_success(
        runs_root=args.runs_root,
        out_dir=args.out_dir,
        stem=args.stem,
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
