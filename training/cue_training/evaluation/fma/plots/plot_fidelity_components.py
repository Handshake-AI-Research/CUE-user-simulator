"""Dot chart of three fidelity and three calibration components.

One row per component, one marker per method, 0–1 with right = better.
User-conditioned arms feed User Mimicry and the three calibration rows. Sampled
arms feed Population Coverage. Naturalness takes the max of the two modes, so a family
with only a user-conditioned arm (PPol) still appears there.

Only Gemini simulators are drawn, so UserLM and USP (Llama-only) are left out
unless `select_series` is given a fallback simulator.

Fidelity:
  Naturalness           mean(Nat-S2R, 1 − 2·Nat-TT)   TT is |0.5 − P(human)|
  User Mimicry          mean(Mim-AVA/100, Mim-PT3)
  Population Coverage   mean(Cov-S2RChamfer, Cov-SDChamfer)

Calibration (inverted so right = better):
  Role Adherence        1 − User Error rate      simulated user stays in role
  Failure Mode Reproduction  1 − TVD_Agent            fails in the same ways as humans
  Success Rate Alignment     1 − |Δ Success rate|     succeeds as often as humans

    uv run --group fma cue-fma components
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

from cue_training.evaluation.fma.compare_fidelity_calibration import (
    Row,
    join_rows,
    load_calibration,
)
from cue_training.evaluation.fma.plots.plot_failure_types import (
    COLORS,
    DPI,
    FAMILY_ORDER,
    FONT_FAMILY,
    MODEL_LABELS,
    PAIRED_SPECS,
    SAMPLED_SPECS,
    _pyplot,
    canonical_family,
)
from cue_training.evaluation.fma.plots.plot_metrics_scatter import (
    load_leaderboard_macro,
)

STEM = "fidelity_calibration_components"
SIMULATOR = "gemini"
FALLBACK_SIMULATOR = None
_EDGE = "#052326"
_GRID = "lightgray"
_FIDELITY_BG = "#ffffff"
_CALIBRATION_BG = (249 / 255, 255 / 255, 230 / 255)

AXIS_LABELS = (
    "Naturalness",
    "User Mimicry",
    "Population Coverage",
    "Role Adherence",
    "Failure Mode Reproduction",
    "Success Rate Alignment",
)
SCORE_KEYS = AXIS_LABELS
N_FIDELITY = 3
SAMPLED_KEYS = frozenset({"Population Coverage"})
MAX_KEYS = frozenset({"Naturalness"})
TINY = 7

MARKERS = {
    "userlm": "o",
    "usp": "s",
    "ppol": "^",
    "realusersim": "D",
    "cue-general": "*",
}


def _mean(values: list[float]) -> float | None:
    return float(np.mean(values)) if values else None


def scale_fidelity(short: str, raw: float) -> float:
    """Map a named fidelity metric onto [0, 1] with higher = better."""

    x = float(raw)
    if short == "Nat-S2R":
        return x
    if short == "Nat-TT":
        return 1.0 - min(max(x, 0.0), 0.5) / 0.5
    if short == "Mim-AVA":
        return x / 100.0
    return x


def component_scores(row: Row) -> dict[str, float | None]:
    """Six component scores; every value is 0–1 with 1 = right edge = better."""

    nat = _mean(
        [
            scale_fidelity(s, row.fidelity[s])
            for s in ("Nat-S2R", "Nat-TT")
            if s in row.fidelity
        ]
    )
    mim = _mean(
        [
            scale_fidelity(s, row.fidelity[s])
            for s in ("Mim-AVA", "Mim-PT3")
            if s in row.fidelity
        ]
    )
    cov = _mean(
        [
            scale_fidelity(s, row.fidelity[s])
            for s in ("Cov-S2RChamfer", "Cov-SDChamfer")
            if s in row.fidelity
        ]
    )
    return {
        "Naturalness": nat,
        "User Mimicry": mim,
        "Population Coverage": cov,
        "Role Adherence": 1.0 - float(row.user_error_pct) / 100.0,
        "Failure Mode Reproduction": 1.0 - float(row.tvd_agent),
        "Success Rate Alignment": 1.0 - float(row.abs_delta_success),
    }


def in_group(family: str, arm: str, specs: set[tuple[str, str | None]]) -> bool:
    if family == "human":
        return False
    return (family, arm) in specs


def is_paired_row(family: str, arm: str) -> bool:
    return (family, arm) in PAIRED_SPECS


def is_sampled_row(family: str, arm: str) -> bool:
    return (family, arm) in SAMPLED_SPECS


def select_series(
    rows: list[Row],
    *,
    simulator: str = SIMULATOR,
    fallback: str | None = FALLBACK_SIMULATOR,
) -> list[Row]:
    """FMA-grouped arms on `simulator` (both modes when they exist)."""

    graded = [
        row
        for row in rows
        if is_paired_row(row.family, row.arm) or is_sampled_row(row.family, row.arm)
    ]
    out: list[Row] = []
    for fam in FAMILY_ORDER:
        pool = [row for row in graded if canonical_family(row.family) == fam]
        chosen = [row for row in pool if row.simulator == simulator] or [
            row for row in pool if fallback and row.simulator == fallback
        ]
        out.extend(
            sorted(chosen, key=lambda r: 0 if is_paired_row(r.family, r.arm) else 1)
        )
    return out


def component_points(series: list[Row]) -> list[tuple[str, Row, float]]:
    """One (component, row, score) per family.

    Population Coverage is sampled-only. Naturalness takes the max over both modes.
    Everything else uses the user-conditioned arm.
    """

    points: list[tuple[str, Row, float]] = []
    for key in SCORE_KEYS:
        for fam in FAMILY_ORDER:
            family_rows = [row for row in series if canonical_family(row.family) == fam]
            if key in MAX_KEYS:
                candidates = family_rows
            else:
                want_sampled = key in SAMPLED_KEYS
                candidates = [
                    row
                    for row in family_rows
                    if is_sampled_row(row.family, row.arm) == want_sampled
                ]
            scored = [
                (row, float(value))
                for row in candidates
                if (value := component_scores(row)[key]) is not None
            ]
            if not scored:
                continue
            if key in MAX_KEYS:
                row, value = max(scored, key=lambda item: item[1])
            else:
                row, value = scored[0]
            points.append((key, row, value))
    return points


def series_label(row: Row, *, simulator: str = SIMULATOR) -> str:
    fam = canonical_family(row.family)
    name = MODEL_LABELS.get(fam, fam)
    if row.simulator == simulator:
        return name
    return f"{name} ({row.simulator.title()})"


def _family_color(family: str) -> str:
    return COLORS.get(MODEL_LABELS.get(family, family), "#888888")


def plot_components(
    rows: list[Row],
    out_dir: Path,
    *,
    stem: str = STEM,
    simulator: str = SIMULATOR,
) -> list[Path]:
    plt = _pyplot()
    from matplotlib.lines import Line2D

    series = select_series(rows, simulator=simulator)
    if not series:
        raise SystemExit(f"no component series for simulator={simulator}")

    n = len(SCORE_KEYS)
    n_fid = N_FIDELITY
    y_fid_header = float(n + 1)
    y_cal_header = float(n - n_fid)
    split = y_cal_header + 0.5

    def y_of(key: str) -> float:
        i = SCORE_KEYS.index(key)
        if i < n_fid:
            return float(n - i)
        return float(n - n_fid - 1 - (i - n_fid))

    fig, ax = plt.subplots(figsize=(4.3, 2.35))
    ax.axhspan(
        split, y_fid_header + 0.55, facecolor=_FIDELITY_BG, edgecolor="none", zorder=0
    )
    ax.axhspan(-0.55, split, facecolor=_CALIBRATION_BG, edgecolor="none", zorder=0)
    ax.text(
        0.5,
        y_fid_header,
        "Fidelity",
        ha="center",
        va="center",
        fontsize=TINY,
        fontweight=600,
        color=_EDGE,
        zorder=2,
    )
    ax.text(
        0.5,
        y_cal_header,
        "Calibration",
        ha="center",
        va="center",
        fontsize=TINY,
        fontweight=600,
        color=_EDGE,
        zorder=2,
    )
    for key, row, value in component_points(series):
        fam = canonical_family(row.family)
        color = _family_color(fam)
        ax.scatter(
            [value],
            [y_of(key)],
            marker=MARKERS.get(fam, "o"),
            s=42 if MARKERS.get(fam) == "*" else 24,
            facecolors=color,
            edgecolors=_EDGE,
            linewidths=0.6,
            zorder=3,
        )

    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(-0.55, y_fid_header + 0.55)
    ax.set_xticks([0.0, 0.25, 0.5, 0.75, 1.0])
    ax.set_xticklabels(["0", "0.25", "0.50", "0.75", "1"], fontsize=TINY, color=_GRID)
    metric_ys = [y_of(k) for k in SCORE_KEYS]
    ax.set_yticks(metric_ys)
    ax.set_yticklabels(list(AXIS_LABELS), fontsize=TINY)
    # Horizontal lines sit on the metric ticks only, so each one traces the
    # single row of markers that are comparable to each other.
    ax.grid(axis="both", color=_GRID, linestyle=":", linewidth=0.5)
    ax.set_axisbelow(True)
    ax.tick_params(axis="y", length=0, pad=2)
    ax.tick_params(axis="x", length=0, pad=1.5, colors=_GRID)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color(_GRID)

    seen: dict[str, Row] = {}
    for row in series:
        seen.setdefault(series_label(row, simulator=simulator), row)
    family_handles = [
        Line2D(
            [0],
            [0],
            linestyle="none",
            marker=MARKERS.get(canonical_family(row.family), "o"),
            markersize=6.5 if MARKERS.get(canonical_family(row.family)) == "*" else 4.5,
            markerfacecolor=_family_color(canonical_family(row.family)),
            markeredgecolor=_EDGE,
            markeredgewidth=0.6,
        )
        for row in seen.values()
    ]
    ax.legend(
        family_handles,
        list(seen),
        loc="lower center",
        bbox_to_anchor=(0.5, 1.0),
        ncol=len(seen),
        fontsize=TINY,
        frameon=False,
        columnspacing=0.8,
        handletextpad=0.3,
        borderpad=0.0,
    )
    ax.set_title(
        "User Simulator Performance on τ²-Bench",
        fontsize=9,
        fontfamily=FONT_FAMILY,
        fontweight=600,
        pad=16,
    )
    fig.subplots_adjust(left=0.52, right=0.99, top=0.84, bottom=0.14)
    out_dir.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []
    for ext in ("png", "pdf"):
        path = out_dir / f"{stem}.{ext}"
        fig.savefig(path, dpi=DPI, bbox_inches="tight", pad_inches=0.05)
        written.append(path)
    plt.close(fig)
    return written


def run_components(
    *,
    leaderboard: Path,
    distributions: Path,
    out_dir: Path,
    stem: str = STEM,
) -> dict[str, Any]:
    lb = load_leaderboard_macro(Path(leaderboard))
    cal = load_calibration(Path(distributions))
    rows, notes = join_rows(lb, cal)
    written = plot_components(rows, Path(out_dir), stem=stem)
    return {
        "n_rows": len(rows),
        "notes": notes,
        "figures": [str(p) for p in written],
    }


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--leaderboard",
        type=Path,
        default=Path("output/metrics/tau2_customer-service/leaderboard.json"),
    )
    parser.add_argument(
        "--distributions",
        type=Path,
        default=Path("fma/distributions.json"),
    )
    parser.add_argument("--out-dir", type=Path, default=Path("fma/figures"))
    parser.add_argument("--stem", default=STEM)
    args = parser.parse_args(argv)
    result = run_components(
        leaderboard=args.leaderboard,
        distributions=args.distributions,
        out_dir=args.out_dir,
        stem=args.stem,
    )
    print(json.dumps({k: v for k, v in result.items() if k != "notes"}, indent=2))
    if result.get("notes"):
        print(f"Notes ({len(result['notes'])}):")
        for line in result["notes"][:20]:
            print(f"  - {line}")


if __name__ == "__main__":
    main()
