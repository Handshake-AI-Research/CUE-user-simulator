"""Compare six fidelity metrics against three calibration metrics.

Fidelity (leaderboard ``domain=macro`` means):
  Nat-S2R, Nat-TT, Mim-AVA, Mim-PT3, Cov-S2RChamfer, Cov-SDChamfer

Calibration (per FMA ``source_id``):
  |Δ Success rate|  = |method success − human success|
  User Error %      = 100 × sim_error_rate from distributions.json
  TVD_Agent         = tvd_agent from distributions.json

Correlations are pairwise-complete: a source missing one fidelity metric is still
used for the others. Coverage metrics (Cov-*) additionally drop paired / user-
conditioned arms (``is_paired``).

Writes a joined points CSV and a Pearson-r correlation matrix (CSV + Markdown).
Optional ``--plot`` draws a 6×3 scatter grid with per-panel r.

    uv run --group fma cue-fma compare \\
        --leaderboard output/metrics/tau2_customer-service/leaderboard.json \\
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

from cue_training.evaluation.fma.plots.plot_metrics_scatter import (
    _SKIP_FAMILIES,
    is_paired,
    load_leaderboard_macro,
    parse_source_id,
    pearson_fit,
)

STEM = "fidelity_vs_calibration"

FIDELITY: tuple[tuple[str, str], ...] = (
    ("classifier/sim2real", "Nat-S2R"),
    ("judge/turing_sonnet_qwen", "Nat-TT"),
    ("mimicry/wegmann_ava", "Mim-AVA"),
    ("mimicry/paired_audit", "Mim-PT3"),
    ("coverage/sim2real_behavioral", "Cov-S2RChamfer"),
    ("coverage/styledistance_behavioral", "Cov-SDChamfer"),
)

# Coverage is population-level; exclude user-conditioned / paired arms.
COVERAGE_SHORTS = frozenset({"Cov-S2RChamfer", "Cov-SDChamfer"})

CALIBRATION: tuple[tuple[str, str], ...] = (
    ("abs_delta_success", "|Δ Success rate|"),
    ("user_error_pct", "User Error %"),
    ("tvd_agent", "TVD_Agent"),
)


@dataclass(frozen=True)
class Row:
    source_id: str
    family: str
    simulator: str
    arm: str
    fidelity: dict[str, float]
    abs_delta_success: float
    user_error_pct: float
    tvd_agent: float
    success: float
    delta_success: float


def load_calibration(path: Path) -> dict[str, dict[str, float]]:
    """``source_id`` → ``{sim_error_rate, tvd_agent}`` from distributions.json."""

    report = json.loads(path.read_text(encoding="utf-8"))
    out: dict[str, dict[str, float]] = {}
    for sid, row in (report.get("comparisons") or {}).items():
        sim_err = row.get("sim_error_rate")
        tvd = row.get("tvd_agent")
        if not isinstance(sim_err, (int, float)) or not isinstance(tvd, (int, float)):
            continue
        out[str(sid)] = {
            "sim_error_rate": float(sim_err),
            "tvd_agent": float(tvd),
        }
    return out


def join_rows(
    leaderboard: dict[tuple[str, str, str], dict[str, float]],
    calibration: dict[str, dict[str, float]],
) -> tuple[list[Row], list[str]]:
    """Join calibration sources to leaderboard; keep partial fidelity values."""

    human_metrics = leaderboard.get(("human", "", "human")) or leaderboard.get(
        ("human", "", "")
    )
    if not human_metrics or "env/tau2_success_rate" not in human_metrics:
        raise ValueError("leaderboard is missing human macro Success rate")
    human_suc = float(human_metrics["env/tau2_success_rate"])

    rows: list[Row] = []
    notes: list[str] = []
    for sid, cal in sorted(calibration.items()):
        parsed = parse_source_id(sid)
        if parsed is None:
            notes.append(f"{sid}: unparseable source_id")
            continue
        family, sim, arm = parsed
        if family in _SKIP_FAMILIES:
            continue
        metrics = leaderboard.get((family, sim, arm))
        if metrics is None:
            notes.append(
                f"{sid}: no macro leaderboard row for ({family}, {sim}, {arm})"
            )
            continue
        suc = metrics.get("env/tau2_success_rate")
        if suc is None:
            notes.append(f"{sid}: missing Success rate")
            continue
        fidelity: dict[str, float] = {}
        for key, short in FIDELITY:
            raw = metrics.get(key)
            if raw is None:
                notes.append(f"{sid}: missing {key} (pairwise skip for {short} only)")
                continue
            fidelity[short] = float(raw)
        if not fidelity:
            notes.append(f"{sid}: no fidelity metrics present")
            continue
        delta = float(suc) - human_suc
        rows.append(
            Row(
                source_id=sid,
                family=family,
                simulator=sim,
                arm=arm,
                fidelity=fidelity,
                abs_delta_success=abs(delta),
                user_error_pct=100.0 * float(cal["sim_error_rate"]),
                tvd_agent=float(cal["tvd_agent"]),
                success=float(suc),
                delta_success=delta,
            )
        )
    return rows, notes


def _calib_of(row: Row, key: str) -> float:
    return float(getattr(row, key))


def fit_rows_for(rows: list[Row], x_short: str) -> list[Row]:
    """Rows eligible for correlating ``x_short`` (present + coverage arm filter)."""

    out: list[Row] = []
    for row in rows:
        if x_short not in row.fidelity:
            continue
        if x_short in COVERAGE_SHORTS and is_paired(row.family, row.arm):
            continue
        out.append(row)
    return out


def correlations(rows: list[Row]) -> tuple[dict[str, float], dict[str, int]]:
    """Pairwise Pearson r and per-cell n, keyed ``<fidelity>_vs_<calib>``."""

    out: dict[str, float] = {}
    ns: dict[str, int] = {}
    for y_key, _ylabel in CALIBRATION:
        for _mkey, x_short in FIDELITY:
            subset = fit_rows_for(rows, x_short)
            key = f"{x_short}_vs_{y_key}"
            ns[key] = len(subset)
            if len(subset) < 3:
                continue
            xs = [r.fidelity[x_short] for r in subset]
            ys = [_calib_of(r, y_key) for r in subset]
            fit = pearson_fit(xs, ys)
            if fit is not None:
                out[key] = round(fit[2], 4)
    return out, ns


def write_points_csv(rows: list[Row], path: Path) -> None:
    fieldnames = [
        "source_id",
        "family",
        "simulator",
        "arm",
        "paired",
        *[short for _k, short in FIDELITY],
        "success",
        "delta_success",
        "abs_delta_success",
        "user_error_pct",
        "tvd_agent",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for r in rows:
            writer.writerow(
                {
                    "source_id": r.source_id,
                    "family": r.family,
                    "simulator": r.simulator,
                    "arm": r.arm,
                    "paired": "1" if is_paired(r.family, r.arm) else "0",
                    **{
                        short: (
                            f"{r.fidelity[short]:.8f}" if short in r.fidelity else ""
                        )
                        for _k, short in FIDELITY
                    },
                    "success": f"{r.success:.8f}",
                    "delta_success": f"{r.delta_success:.8f}",
                    "abs_delta_success": f"{r.abs_delta_success:.8f}",
                    "user_error_pct": f"{r.user_error_pct:.8f}",
                    "tvd_agent": f"{r.tvd_agent:.8f}",
                }
            )


def write_corr_csv(corr: dict[str, float], ns: dict[str, int], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "fidelity",
                *[f"{y}_r" for _k, y in CALIBRATION],
                *[f"{y}_n" for _k, y in CALIBRATION],
            ]
        )
        for _mkey, x_short in FIDELITY:
            r_cells = []
            n_cells = []
            for y_key, _ylab in CALIBRATION:
                key = f"{x_short}_vs_{y_key}"
                r_cells.append(f"{corr[key]:.4f}" if key in corr else "")
                n_cells.append(str(ns.get(key, 0)))
            writer.writerow([x_short, *r_cells, *n_cells])


def format_corr_markdown(corr: dict[str, float], ns: dict[str, int]) -> str:
    headers = ["Fidelity", *[ylab for _k, ylab in CALIBRATION]]
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    for _mkey, x_short in FIDELITY:
        cells = [x_short]
        for y_key, _ylab in CALIBRATION:
            key = f"{x_short}_vs_{y_key}"
            n = ns.get(key, 0)
            if key not in corr:
                cells.append(f"— (n={n})")
            else:
                cells.append(f"{corr[key]:.3f} (n={n})")
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines) + "\n"


def plot_grid(rows: list[Row], out_dir: Path, *, stem: str) -> list[Path]:
    from cue_training.evaluation.fma.plots.plot_failure_types import (
        DPI,
        FONT_FAMILY,
        _pyplot,
    )
    from cue_training.evaluation.fma.plots.plot_metrics_scatter import (
        _EDGE,
        _FIT_COLOR,
        _MARKER_BY_SIM,
        color_of,
        edge_style,
    )

    plt = _pyplot()
    fig, axes = plt.subplots(
        nrows=len(CALIBRATION),
        ncols=len(FIDELITY),
        figsize=(12.0, 5.5),
        sharey="row",
    )
    for r_i, (y_key, ylabel) in enumerate(CALIBRATION):
        for c_i, (_mkey, x_short) in enumerate(FIDELITY):
            ax = axes[r_i][c_i]
            subset = fit_rows_for(rows, x_short)
            if not subset:
                ax.set_axis_off()
                continue
            xs = [row.fidelity[x_short] for row in subset]
            ys = [_calib_of(row, y_key) for row in subset]
            ymin, ymax = min(ys), max(ys)
            for row in subset:
                ax.scatter(
                    [row.fidelity[x_short]],
                    [_calib_of(row, y_key)],
                    s=70,
                    c=color_of(row.family),
                    marker=_MARKER_BY_SIM.get(row.simulator, "o"),
                    edgecolors=_EDGE,
                    linewidths=1.1,
                    linestyles=edge_style(row.family, row.arm),
                    zorder=3,
                    clip_on=False,
                )
            fit = pearson_fit(xs, ys)
            if fit is not None:
                slope, intercept, pearson_r = fit
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
                    f"r = {pearson_r:.2f} (n={len(subset)})",
                    transform=ax.transAxes,
                    fontsize=6.5,
                    va="top",
                    ha="left",
                    color=_EDGE,
                )
            if r_i == len(CALIBRATION) - 1:
                ax.set_xlabel(x_short, fontsize=8)
            if c_i == 0:
                ax.set_ylabel(ylabel, fontsize=8)
            ax.grid(True, alpha=0.25, linewidth=0.6)
            pad = max(0.08 * max(ymax - ymin, 1e-6), 0.02 * max(abs(ymax), 1.0))
            ax.set_ylim(ymin - pad, ymax + pad)
            ax.tick_params(labelsize=6.5)

    fig.suptitle(
        "Fidelity vs calibration (pairwise; Cov-* exclude paired)",
        fontsize=10,
        fontfamily=FONT_FAMILY,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    out_dir.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []
    for ext in ("png", "pdf"):
        path = out_dir / f"{stem}.{ext}"
        fig.savefig(path, dpi=DPI, bbox_inches="tight")
        written.append(path)
    plt.close(fig)
    return written


def run_compare(
    *,
    leaderboard: Path,
    distributions: Path,
    out_dir: Path,
    stem: str = STEM,
    plot: bool = False,
) -> dict[str, Any]:
    lb = load_leaderboard_macro(Path(leaderboard))
    cal = load_calibration(Path(distributions))
    rows, missing = join_rows(lb, cal)
    if not rows:
        raise ValueError("No joined rows; check leaderboard / distributions mapping")

    out = Path(out_dir)
    corr, ns = correlations(rows)
    points_path = out / f"{stem}_points.csv"
    corr_csv = out / f"{stem}_correlations.csv"
    corr_md = out / f"{stem}_correlations.md"
    write_points_csv(rows, points_path)
    write_corr_csv(corr, ns, corr_csv)
    corr_md.write_text(format_corr_markdown(corr, ns), encoding="utf-8")

    figures: list[str] = []
    if plot:
        figures = [str(p) for p in plot_grid(rows, out, stem=stem)]

    return {
        "n_rows": len(rows),
        "correlations": corr,
        "n_per_cell": ns,
        "missing": missing,
        "points_csv": str(points_path),
        "correlations_csv": str(corr_csv),
        "correlations_md": str(corr_md),
        "figures": figures,
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
    parser.add_argument(
        "--plot",
        action="store_true",
        help="Also write a 6×3 scatter PNG/PDF with per-panel Pearson r",
    )
    args = parser.parse_args(argv)
    result = run_compare(
        leaderboard=args.leaderboard,
        distributions=args.distributions,
        out_dir=args.out_dir,
        stem=args.stem,
        plot=bool(args.plot),
    )
    if result.get("missing"):
        print(f"Skipped {len(result['missing'])} source(s):")
        for line in result["missing"]:
            print(f"  - {line}")
    print(Path(result["correlations_md"]).read_text(encoding="utf-8"))
    print(
        json.dumps(
            {k: v for k, v in result.items() if k != "missing"},
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
