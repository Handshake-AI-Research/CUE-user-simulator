"""Six-panel metric scatters: naturalness / mimicry / coverage × ΔSuc / TVD-agent.

Style matches ``failure-mode-analysis/fma/plot_naturalness_tvd.py``:
  fill = family, shape = simulator, edge linestyle = conditioning.

Each panel carries a least-squares line and its Pearson r, fit over the simulator points
only (see ``fit_points``). r is printed per panel and returned in the run summary.

X metrics (from aggregate leaderboard ``domain=macro``):
  Naturalness = Sim2Real P(human)   (``classifier/sim2real``)
  Mimicry     = AVA / 100           (``mimicry/wegmann_ava``, an extended-suite metric)
  Coverage    = SD coverage         (``coverage/styledistance_behavioral``)

Y metrics:
  Δ Suc vs Human = method success − human success (leaderboard Success rate)
  TVD agent      = agent-only failure-mix TVD from ``distributions.json``

    uv run --group fma cue-fma scatter \\
        --leaderboard extended-output/metrics/tau2_customer-service/leaderboard.json \\
        --distributions fma/distributions.json \\
        --out-dir fma/figures
"""

from __future__ import annotations

import argparse
import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from cue_training.evaluation.fma.plots.plot_failure_types import (
    COLORS,
    DPI,
    MODEL_LABELS,
    PAIRED_SPECS,
    SAMPLED_SPECS,
    _pyplot,
)

FIGSIZE = (7, 3)
STEM = "metrics_vs_delta_suc_tvd"
_LEGEND_X = 0.87
_Y_PAD = 0.08
_SKIP_FAMILIES = frozenset({"base", "baseline-base-once"})
_SIMS = ("llama", "gpt", "gemini")
_MARKER_BY_SIM = {"llama": "o", "gpt": "s", "gemini": "^"}
_LABEL_BY_SIM = {"llama": "Llama", "gpt": "GPT", "gemini": "Gemini"}
_EDGE = "#052326"
_FIT_COLOR = "#d62728"
_EDGE_STYLE_PAIRED = "solid"
_EDGE_STYLE_SAMPLED = ":"
_EXTRA_COLORS = {
    "CUE": "#8b5cf0",
    "CUE Refined": "#d4b3ff",
}

# Leaderboard metric keys / display.
X_METRICS: tuple[tuple[str, str, str, float], ...] = (
    # (key, xlabel, short, scale) — scale multiplies raw mean (AVA is 0–100).
    ("classifier/sim2real", "Naturalness (S2R)", "naturalness", 1.0),
    ("mimicry/wegmann_ava", "Mimicry (AVA)", "mimicry", 0.01),
    ("coverage/styledistance_behavioral", "Coverage (SD)", "coverage", 1.0),
)
Y_METRICS: tuple[tuple[str, str, bool, bool], ...] = (
    # (key, ylabel, invert_y, zero_line)
    ("delta_success", "Δ Suc vs Human", False, True),
    ("tvd_agent", "TVD agent ↓", True, False),
)


@dataclass(frozen=True)
class Point:
    source_id: str
    family: str
    simulator: str
    arm: str
    x: dict[str, float]
    delta_success: float
    tvd_agent: float
    success: float


def parse_source_id(source_id: str) -> tuple[str, str, str] | None:
    """``cue-general-gemini|paired`` → (family, sim, arm)."""

    if "|" not in source_id:
        return None
    left, arm = source_id.rsplit("|", 1)
    arm = arm.strip()
    left = left.strip()
    for sim in _SIMS:
        suffix = f"-{sim}"
        if left.endswith(suffix):
            return left[: -len(suffix)], sim, arm
    return None


def family_label(family: str) -> str:
    return MODEL_LABELS.get(family) or family


def color_of(family: str) -> str:
    name = family_label(family)
    if name in _EXTRA_COLORS:
        return _EXTRA_COLORS[name]
    return COLORS.get(name) or "#888888"


def is_paired(family: str, arm: str) -> bool:
    if (family, arm) in PAIRED_SPECS:
        return True
    if (family, arm) in SAMPLED_SPECS:
        return False
    # Fallback: paired / as_is for ppol-style; sampled otherwise.
    return arm in {"paired"}


def edge_style(family: str, arm: str, *, human: bool = False) -> str:
    if human:
        return _EDGE_STYLE_PAIRED
    return _EDGE_STYLE_PAIRED if is_paired(family, arm) else _EDGE_STYLE_SAMPLED


def load_leaderboard_macro(path: Path) -> dict[tuple[str, str, str], dict[str, float]]:
    """Map (family, simulator, arm) → metric means for ``domain=macro`` rows."""

    payload = json.loads(path.read_text(encoding="utf-8"))
    rows = payload.get("rows") if isinstance(payload, dict) else payload
    out: dict[tuple[str, str, str], dict[str, float]] = {}
    for row in rows or []:
        if str(row.get("domain") or "") != "macro":
            continue
        family = str(row.get("family") or row.get("method") or "").strip()
        sim = str(row.get("simulator") or "").strip()
        if sim in {"—", "-", "None", ""}:
            sim = ""
        arm = str(row.get("arm") or "").strip()
        metrics = row.get("metrics") or {}
        means: dict[str, float] = {}
        for key, agg in metrics.items():
            if isinstance(agg, dict) and agg.get("mean") is not None:
                try:
                    means[str(key)] = float(agg["mean"])
                except (TypeError, ValueError):
                    continue
        if means:
            out[(family, sim, arm)] = means
    return out


def load_tvd_agent(path: Path) -> dict[str, float]:
    report = json.loads(path.read_text(encoding="utf-8"))
    out: dict[str, float] = {}
    for sid, row in (report.get("comparisons") or {}).items():
        tvd = row.get("tvd_agent")
        if isinstance(tvd, (int, float)):
            out[str(sid)] = float(tvd)
    return out


def join_points(
    leaderboard: dict[tuple[str, str, str], dict[str, float]],
    tvd_by_source: dict[str, float],
    *,
    include_human: bool = True,
) -> tuple[list[Point], list[str]]:
    human_metrics = leaderboard.get(("human", "", "human")) or leaderboard.get(
        ("human", "", "")
    )
    if not human_metrics or "env/tau2_success_rate" not in human_metrics:
        raise ValueError("leaderboard is missing human macro Success rate")
    human_suc = float(human_metrics["env/tau2_success_rate"])

    points: list[Point] = []
    missing: list[str] = []
    for sid, tvd in sorted(tvd_by_source.items()):
        parsed = parse_source_id(sid)
        if parsed is None:
            missing.append(f"{sid}: unparseable source_id")
            continue
        family, sim, arm = parsed
        if family in _SKIP_FAMILIES:
            continue
        metrics = leaderboard.get((family, sim, arm))
        if metrics is None:
            # Older leaderboards used method=general under family=cue-general already;
            # also try method folder names as family.
            missing.append(
                f"{sid}: no macro leaderboard row for ({family}, {sim}, {arm})"
            )
            continue
        suc = metrics.get("env/tau2_success_rate")
        if suc is None:
            missing.append(f"{sid}: missing Success rate")
            continue
        xvals: dict[str, float] = {}
        ok = True
        for key, _label, short, scale in X_METRICS:
            raw = metrics.get(key)
            if raw is None:
                missing.append(f"{sid}: missing {key}")
                ok = False
                break
            xvals[short] = float(raw) * scale
        if not ok:
            continue
        points.append(
            Point(
                source_id=sid,
                family=family,
                simulator=sim,
                arm=arm,
                x=xvals,
                delta_success=float(suc) - human_suc,
                tvd_agent=float(tvd),
                success=float(suc),
            )
        )

    if include_human:
        xvals = {}
        for key, _label, short, scale in X_METRICS:
            raw = human_metrics.get(key)
            if raw is None:
                raise ValueError(f"human row missing {key}")
            xvals[short] = float(raw) * scale
        points.append(
            Point(
                source_id="human",
                family="human",
                simulator="",
                arm="human",
                x=xvals,
                delta_success=0.0,
                tvd_agent=0.0,
                success=human_suc,
            )
        )
    return points, missing


def _y_of(point: Point, y_key: str) -> float:
    return float(getattr(point, y_key))


def fit_points(points: list[Point]) -> list[Point]:
    """Points the correlation is computed over.

    Human is excluded: it is the reference, pinned to delta_success=0 and tvd_agent=0 by
    construction, so including it would put a fixed point on both trend lines.
    """

    return [p for p in points if p.family != "human"]


def pearson_fit(xs: list[float], ys: list[float]) -> tuple[float, float, float] | None:
    """``(slope, intercept, r)`` for a least-squares line, or None if undefined."""

    import numpy as np

    x = np.asarray(xs, dtype=float)
    y = np.asarray(ys, dtype=float)
    if x.size < 3 or np.ptp(x) == 0 or np.ptp(y) == 0:
        return None
    slope, intercept = (float(v) for v in np.polyfit(x, y, 1))
    return slope, intercept, float(np.corrcoef(x, y)[0, 1])


def correlations(points: list[Point]) -> dict[str, float]:
    """Pearson r per panel, keyed ``<x_short>_vs_<y_key>``, for the run summary."""

    fitted = fit_points(points)
    out: dict[str, float] = {}
    for y_key, _ylabel, _invert, _zero in Y_METRICS:
        for _mkey, _xlabel, x_short, _scale in X_METRICS:
            fit = pearson_fit(
                [p.x[x_short] for p in fitted], [_y_of(p, y_key) for p in fitted]
            )
            if fit is not None:
                out[f"{x_short}_vs_{y_key}"] = round(fit[2], 4)
    return out


def plot_grid(points: list[Point], out_dir: Path, *, stem: str = STEM) -> list[Path]:
    plt = _pyplot()
    from matplotlib.lines import Line2D

    fig, axes = plt.subplots(
        nrows=len(Y_METRICS),
        ncols=len(X_METRICS),
        figsize=FIGSIZE,
        sharex="col",
        sharey="row",
    )

    drawn_families: set[str] = set()
    for r, (y_key, ylabel, invert_y, zero_line) in enumerate(Y_METRICS):
        ys = [_y_of(p, y_key) for p in points]
        ymin, ymax = min(ys), max(ys)
        for c, (_mkey, xlabel, x_short, _scale) in enumerate(X_METRICS):
            ax = axes[r][c]
            for p in points:
                human = p.family == "human"
                style = edge_style(p.family, p.arm, human=human)
                marker = "*" if human else _MARKER_BY_SIM.get(p.simulator, "o")
                ax.scatter(
                    [p.x[x_short]],
                    [_y_of(p, y_key)],
                    s=180 if human else 70,
                    c="#052326" if human else color_of(p.family),
                    marker=marker,
                    edgecolors=_EDGE,
                    linewidths=1.4 if (human or style == _EDGE_STYLE_SAMPLED) else 1.1,
                    linestyles=style,
                    zorder=3,
                    clip_on=False,
                )
                drawn_families.add(family_label(p.family))

            fitted = fit_points(points)
            xs = [p.x[x_short] for p in fitted]
            fit = pearson_fit(xs, [_y_of(p, y_key) for p in fitted])
            if fit is not None:
                slope, intercept, pearson_r = fit
                # axline is infinite and clipped to the axes, so the trend spans the panel
                # regardless of where set_xlim lands below. Note this extrapolates past the
                # observed x range, which is narrow for mimicry and coverage.
                ax.axline(
                    (0.0, intercept),
                    slope=slope,
                    color=_FIT_COLOR,
                    linewidth=1.2,
                    linestyle=(0, (4, 2)),
                    alpha=0.9,
                    zorder=2,
                )
                ax.text(
                    0.03,
                    0.96,
                    f"r = {pearson_r:.2f}",
                    transform=ax.transAxes,
                    fontsize=6.5,
                    va="top",
                    ha="left",
                    color=_EDGE,
                )

            if r == len(Y_METRICS) - 1:
                ax.set_xlabel(xlabel, fontsize=9)
            if c == 0:
                ax.set_ylabel(ylabel, fontsize=9)
            ax.grid(True, alpha=0.25, linewidth=0.6)
            ax.set_xlim(0.0, 1.0 + _Y_PAD)
            if invert_y:
                ax.set_ylim(-_Y_PAD, max(1.0, ymax + _Y_PAD))
                ax.invert_yaxis()
            else:
                span = max(ymax - ymin, 0.1)
                pad = max(_Y_PAD, 0.08 * span)
                ax.set_ylim(ymin - pad, ymax + pad)
                # Percent-style ticks for ΔSuc (every 20% to reduce crowding).
                ticks = [-0.4, -0.2, 0.0, 0.2]
                labels = ["-40%", "-20%", "=", "+20%"]
                lo, hi = ax.get_ylim()
                keep = [(t, lab) for t, lab in zip(ticks, labels) if lo <= t <= hi]
                if keep:
                    ax.set_yticks([t for t, _ in keep], [lab for _, lab in keep])
            if zero_line:
                ax.axhline(
                    0.0, color="#000000", linewidth=1.0, solid_capstyle="butt", zorder=1
                )
                ax.spines["bottom"].set_visible(False)
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)
            ax.tick_params(axis="both", which="both", length=0, labelsize=8)

    family_order = (
        "Human",
        "UserLM",
        "USP",
        "PPol",
        "RealUserSim",
        "CUE",
        "CUE Refined",
    )
    family_handles = []
    for name in family_order:
        if name not in drawn_families:
            continue
        face = (
            "#052326"
            if name == "Human"
            else (_EXTRA_COLORS.get(name) or COLORS.get(name) or "#888888")
        )
        family_handles.append(
            Line2D(
                [0],
                [0],
                marker="o",
                color="none",
                markerfacecolor=face,
                markeredgecolor=_EDGE,
                markersize=7,
                label=name,
            )
        )
    sims_present = {p.simulator for p in points if p.simulator}
    sim_handles = [
        Line2D(
            [0],
            [0],
            marker=_MARKER_BY_SIM[sim],
            color="none",
            markerfacecolor="#dddddd",
            markeredgecolor=_EDGE,
            markersize=7,
            label=_LABEL_BY_SIM[sim],
        )
        for sim in _SIMS
        if sim in sims_present
    ]
    cond_handles = [
        axes[0][0].scatter(
            [],
            [],
            s=50,
            c="#dddddd",
            marker="o",
            edgecolors=_EDGE,
            linewidths=1.4,
            linestyles=_EDGE_STYLE_PAIRED,
            label="User-Cond",
        ),
        axes[0][0].scatter(
            [],
            [],
            s=50,
            c="#dddddd",
            marker="o",
            edgecolors=_EDGE,
            linewidths=1.4,
            linestyles=_EDGE_STYLE_SAMPLED,
            label="Sampled",
        ),
    ]

    leg_kw = {
        "frameon": False,
        "fontsize": 7,
        "borderaxespad": 0.0,
        "handletextpad": 0.35,
    }
    # Stack the three legends outside the grid on the right.
    fig.legend(
        handles=family_handles,
        loc="upper left",
        bbox_to_anchor=(_LEGEND_X, 1.0),
        **leg_kw,
    )
    fig.legend(
        handles=sim_handles,
        loc="upper left",
        bbox_to_anchor=(_LEGEND_X, 0.44),
        **leg_kw,
    )
    fig.legend(
        handles=cond_handles,
        loc="upper left",
        bbox_to_anchor=(_LEGEND_X, 0.18),
        **leg_kw,
    )
    fig.subplots_adjust(
        left=0.07, right=0.86, top=0.97, bottom=0.14, wspace=0.18, hspace=0.22
    )

    out_dir.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []
    for ext in ("png", "pdf"):
        path = out_dir / f"{stem}.{ext}"
        fig.savefig(path, dpi=DPI, bbox_inches="tight")
        written.append(path)
    plt.close(fig)
    return written


def write_csv(points: list[Point], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "source_id",
        "family",
        "simulator",
        "arm",
        "naturalness",
        "mimicry",
        "coverage",
        "success",
        "delta_success",
        "tvd_agent",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for p in points:
            writer.writerow(
                {
                    "source_id": p.source_id,
                    "family": p.family,
                    "simulator": p.simulator,
                    "arm": p.arm,
                    "naturalness": f"{p.x['naturalness']:.8f}",
                    "mimicry": f"{p.x['mimicry']:.8f}",
                    "coverage": f"{p.x['coverage']:.8f}",
                    "success": f"{p.success:.8f}",
                    "delta_success": f"{p.delta_success:.8f}",
                    "tvd_agent": f"{p.tvd_agent:.8f}",
                }
            )


def run_scatter(
    *,
    leaderboard: Path,
    distributions: Path,
    out_dir: Path,
    stem: str = STEM,
    no_human: bool = False,
    no_plot: bool = False,
) -> dict[str, Any]:
    lb = load_leaderboard_macro(Path(leaderboard))
    tvd = load_tvd_agent(Path(distributions))
    points, missing = join_points(lb, tvd, include_human=not no_human)
    if not points:
        raise ValueError("No joined points; check leaderboard / distributions mapping")
    out = Path(out_dir)
    csv_path = out / f"{stem}.csv"
    write_csv(points, csv_path)
    written = [] if no_plot else [str(p) for p in plot_grid(points, out, stem=stem)]
    return {
        "n_points": len(points),
        "n_fit_points": len(fit_points(points)),
        "correlations": correlations(points),
        "missing": missing,
        "csv": str(csv_path),
        "figures": written,
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
    parser.add_argument("--no-human", action="store_true")
    parser.add_argument("--no-plot", action="store_true")
    args = parser.parse_args(argv)
    result = run_scatter(
        leaderboard=args.leaderboard,
        distributions=args.distributions,
        out_dir=args.out_dir,
        stem=args.stem,
        no_human=args.no_human,
        no_plot=args.no_plot,
    )
    if result["missing"]:
        print(f"Skipped {len(result['missing'])} source(s):")
        for line in result["missing"]:
            print(f"  - {line}")
    print(f"Wrote {result['csv']} ({result['n_points']} points)")
    if result["correlations"]:
        print(f"Pearson r over {result['n_fit_points']} non-human points:")
        for panel, value in result["correlations"].items():
            print(f"  {panel}: {value:+.2f}")
    for path in result["figures"]:
        print(f"Wrote {path}")


if __name__ == "__main__":
    main()
