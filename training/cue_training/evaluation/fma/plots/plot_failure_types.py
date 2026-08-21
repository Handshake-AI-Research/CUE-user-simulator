"""Percent-combined failure-mix figure from an FMA run's ``tagged.jsonl``.

Port of ``failure-mode-analysis/fma/plot_failure_types.py`` ``plot_combined_percent``,
adapted to CUE source ids (``{method}-{sim}|{arm}``) and the paper's
paired vs sampled arm split:

  Paired / user-conditioned:  cue-general paired, usp paired, ppol as_is,
                              realusersim_paired_noex paired (shown as RealUserSim)
  Sampled:                    cue-general sampled_uncond, userlm, usp sample_diverse,
                              realusersim as_is (shown as RealUserSim)

Install the FMA dependencies with ``uv sync --group fma``:

    uv run --group fma cue-fma plot --run "$RUN"
"""

from __future__ import annotations

import textwrap
from collections import Counter
from collections.abc import Iterable
from pathlib import Path
from typing import Any

import numpy as np

from cue_training.evaluation.common.io import read_jsonl
from cue_training.evaluation.fma.paths import tagged_path
from cue_training.evaluation.fma.sim_errors import (
    ENVIRONMENT_ERROR_LABEL,
    LEGACY_SIMULATOR_ERROR_LABEL,
    PAPER_CUT_MERGE,
    USER_ERROR_BUCKET_SOURCES,
    USER_ERROR_BUCKET_TARGET,
)
from cue_training.evaluation.fma.store import load_taxonomy

REFERENCE_MODEL = "human"
BASE_MODELS = ("gemini", "gpt", "llama")
NO_BASE_MODEL = "n/a"

DPI = 150
LABEL_WIDTH = 18
FONT_FAMILY = "Inter Tight"
FONT_WEIGHTS = (400, 600, 700)

COMBINED_FIGSIZE = (7.5, 3.2)
DETAIL_LABEL_ROTATION = 35
AGENT_LABEL_ROTATION = 0
# Alternating band behind every other category (bars + its x label) to group the columns.
# Same wash as the calibration-row background in ``plot_fidelity_components``.
CATEGORY_BAND_COLOR = "#eeeef1"
# Figure-fraction padding so the stripe sits a little above the top grid and a
# little below both lines of the category label instead of flush with them.
CATEGORY_BAND_PAD = 0.014
COMBINED_SHARE_Y = False
# Bars are clipped at the cap and annotated with their true value, so one runaway
# arm cannot flatten every other series.
COMBINED_PERCENT_YLIM = (0, 50)
COMBINED_YTICK_STEP = 25
AGENT_PERCENT_YLIM = (0, 30)
AGENT_YTICK_STEP = 15
# Fraction of each category slot filled by the grouped bars (agent-only plot is wider).
COMBINED_BAR_SPAN = 0.8
AGENT_BAR_SPAN = 0.72
# Companion panel: User / Agent / Env share of all failures.
SUMMARY_PERCENT_YLIM = (0, 100)
SUMMARY_YTICK_STEP = 50
SUMMARY_WIDTH_RATIO = 0.7
DETAIL_WIDTH_RATIO = 4
SMALL, MEDIUM = 9, 11
LABEL_MIX = 0.45

# Taxonomy labels folded into the summary User / Env buckets (not on the detail axis).
USER_ERROR_TAXONOMY_LABELS = frozenset(
    {"Other Simulator Error", LEGACY_SIMULATOR_ERROR_LABEL}
)
ENV_ERROR_TAXONOMY_LABELS = frozenset({ENVIRONMENT_ERROR_LABEL})
SUMMARY_TYPES = ("User Error", "Agent Error", "Env Error")
SUMMARY_LABELS = {
    "User Error": "User",
    "Agent Error": "Agent",
    "Env Error": "Env",
}

COLORS = {
    "Human": "#052326",
    "UserLM": "#d3fb52",
    "USP": "#7af3ff",
    "PPol": "#2e6aff",
    "RealUserSim": "#a6f7a8",
    "CUE": "#bf94f7",
    "CUE Refined": "#dcb3ff",
}

MODEL_LABELS = {
    "human": "Human",
    "userlm": "UserLM",
    "usp": "USP",
    "ppol": "PPol",
    "realusersim": "RealUserSim",
    # Paired-noex is the conditioned RealUserSim arm; same family color/legend as as_is.
    "realusersim_paired_noex": "RealUserSim",
    "cue-general": "CUE",
}

# Hatch encodes conditioning. Switched vs the old FMA defaults: PPol as_is is
# paired/user-conditioned here; RealUserSim as_is is sampled; paired-noex is
# the user-conditioned RealUserSim bar (same RealUserSim label, paired hatch).
PAIRED_SPECS = {
    ("cue-general", "paired"),
    ("usp", "paired"),
    ("ppol", "as_is"),
    ("realusersim_paired_noex", "paired"),
}
SAMPLED_SPECS = {
    ("cue-general", "sampled_uncond"),
    ("userlm", "as_is"),
    ("userlm", None),
    ("usp", "sample_diverse"),
    ("realusersim", "as_is"),
}
PAIRED_HATCH = ""
SAMPLED_HATCH = "///////"
CONDITIONING_LEGEND = (
    ("User-conditioned", PAIRED_HATCH),
    ("Sampled user", SAMPLED_HATCH),
)

# Short x-axis labels for the current taxonomy; unknown modes wrap the raw name.
FAILURE_LABELS = {
    "User Data Leakage": "Data\nLeakage",
    "Unnecessary Escalation": "Unneeded\nTransfer",
    "Ignoring/Not Gathering Available Information": "Ignored\nInfo",
    "Wrong Action Parameter": "TC Wrong\nParam",
    "Policy-Forbidden Action": "Policy\nViolation",
    "Authentication Deadlock": "Auth\nDeadlock",
    "Failure to Provide Correct Options": "Wrong\nOptions",
    "Premature Action": "Premature\nTC",
    "Unrequested Action": "Unasked\nTC",
    "Incorrect Policy Guidance": "Bad Policy\nAdvice",
    "Premature User Stop": "Early\nStop",
    "User Identity / Task Derailment": "Identity\nDerail",
    # Default paper cut: Identity + Leakage fold into this; shown as User Error.
    "Other Simulator Error": "User\nError",
    "Critical User Simulator Error": "User\nError",
    "Environment Error": "Env\nError",
    "Unexecuted Action After Confirmation": "Missed\nTC",
    "Incorrect Payment Method or Amount": "Payment\nError",
    "Uncategorized": "Uncategorized",
}

MODEL_NAMES = {
    "llama": "Llama 3.1 8B",
    "gpt": "GPT 5.4 Mini",
    "gemini": "Gemini 3.5 Flash Lite",
}

FAMILY_ORDER = (
    "userlm",
    "usp",
    "ppol",
    "realusersim",
    "cue-general",
    "cue-refined",
)


def canonical_family(family: str) -> str:
    """Collapse method variants that share one paper family (label + legend)."""

    if family == "realusersim_paired_noex":
        return "realusersim"
    return family


def parse_source(source_id: str) -> tuple[str, str | None, str | None]:
    """Return ``(family, simulator, arm)`` for a tagged ``source_id``."""

    sid = str(source_id or "").strip()
    if sid == "human" or not sid:
        return ("human", None, None)
    body, _, arm = sid.partition("|")
    arm_val = arm or None
    for sim in BASE_MODELS:
        suffix = f"-{sim}"
        if body.endswith(suffix):
            return (body[: -len(suffix)], sim, arm_val)
    return (body, None, arm_val)


def is_included(source_id: str) -> bool:
    family, _sim, arm = parse_source(source_id)
    if family == "human":
        return True
    key = (family, arm)
    return key in PAIRED_SPECS or key in SAMPLED_SPECS


def hatch_of(source_id: str) -> str:
    family, _sim, arm = parse_source(source_id)
    if family == "human":
        return ""
    key = (family, arm)
    if key in SAMPLED_SPECS:
        return SAMPLED_HATCH
    if key in PAIRED_SPECS:
        return PAIRED_HATCH
    return ""


def label_of(source_id: str) -> str:
    family, _sim, _arm = parse_source(source_id)
    return MODEL_LABELS.get(family, family)


def base_model_of(source_id: str) -> str:
    _family, sim, _arm = parse_source(source_id)
    return sim if sim in BASE_MODELS else NO_BASE_MODEL


def sort_key(source_id: str) -> tuple[int, int, str]:
    family, _sim, arm = parse_source(source_id)
    base = base_model_of(source_id)
    base_order = list(BASE_MODELS) + [NO_BASE_MODEL]
    fam_order = list(FAMILY_ORDER)
    # Paired-noex shares RealUserSim's family slot; arm ranks conditioned before sampled.
    arm_rank = {
        "paired": "0",
        "as_is": "1",
        "sample_diverse": "1",
        "sampled_uncond": "1",
    }
    fam = canonical_family(family)
    return (
        base_order.index(base) if base in base_order else len(base_order),
        fam_order.index(fam) if fam in fam_order else len(fam_order),
        arm_rank.get(arm or "", arm or ""),
    )


def load_counts(
    run_dir: Path,
    *,
    exclude_labels: Iterable[str] | None = None,
    merge_labels: dict[str, str] | None = None,
) -> tuple[dict[str, Counter], list[str]]:
    """Per-source label counts from ``tagged.jsonl``, filtered to the figure series."""

    path = tagged_path(run_dir)
    if not path.is_file():
        raise FileNotFoundError(
            f"missing tagged.jsonl under {run_dir}; run `cue-fma tag` first"
        )
    merge = {
        str(k).strip(): str(v).strip()
        for k, v in (merge_labels or {}).items()
        if str(k).strip() and str(v).strip() and str(k).strip() != str(v).strip()
    }
    drop = {str(x).strip() for x in (exclude_labels or []) if str(x).strip()}
    drop |= set(merge)
    counts: dict[str, Counter] = {}
    for row in read_jsonl(str(path)):
        sid = str(row.get("source_id") or "")
        if not is_included(sid):
            continue
        label = str(row.get("label") or "Uncategorized").strip() or "Uncategorized"
        label = merge.get(label, label)
        if label in drop:
            continue
        counts.setdefault(sid, Counter())[label] += 1
    if not counts:
        raise SystemExit(f"no included tagged rows under {run_dir}")

    taxonomy = [m["name"] for m in load_taxonomy(run_dir) if m.get("name")]
    observed = {t for c in counts.values() for t in c}
    # Keep every taxonomy mode (minus merged/excluded ones) so the x-axis is stable
    # across cuts, even when a filter leaves a mode with no surviving rows.
    types = [t for t in taxonomy if t not in drop] + sorted(observed - set(taxonomy))
    # Prefer FAILURE_LABELS order when present (overridden by human-frequency sort in run_plot).
    preferred = [t for t in FAILURE_LABELS if t in types]
    types = preferred + [t for t in types if t not in preferred]
    return counts, types


def order_types_by_human(counts: dict[str, Counter], types: list[str]) -> list[str]:
    """X-axis order: most common human failure first; User Error / Env Error last."""

    trailing = (
        "Other Simulator Error",
        LEGACY_SIMULATOR_ERROR_LABEL,
        ENVIRONMENT_ERROR_LABEL,
    )
    trail_rank = {name: i for i, name in enumerate(trailing)}
    human = counts.get(REFERENCE_MODEL) or Counter()

    def key(t: str) -> tuple[int, int, str]:
        if t in trail_rank:
            return (1, trail_rank[t], t)
        return (0, -int(human.get(t, 0)), t)

    return sorted(types, key=key)


def agent_detail_types(types: list[str]) -> list[str]:
    skip = USER_ERROR_TAXONOMY_LABELS | ENV_ERROR_TAXONOMY_LABELS
    return [t for t in types if t not in skip]


def as_percent(counts: dict[str, Counter]) -> dict[str, Counter]:
    shares: dict[str, Counter] = {}
    for model, values in counts.items():
        total = sum(values.values())
        shares[model] = (
            Counter({k: 100.0 * v / total for k, v in values.items()})
            if total
            else Counter()
        )
    return shares


def as_agent_percent(
    counts: dict[str, Counter], agent_types: list[str]
) -> dict[str, Counter]:
    """Agent-mode shares with agent failures only as the denominator."""

    shares: dict[str, Counter] = {}
    for model, values in counts.items():
        total = sum(int(values.get(t, 0)) for t in agent_types)
        shares[model] = (
            Counter({t: 100.0 * int(values.get(t, 0)) / total for t in agent_types})
            if total
            else Counter({t: 0.0 for t in agent_types})
        )
    return shares


def as_bucket_percent(counts: dict[str, Counter]) -> dict[str, Counter]:
    """Collapse failures into User / Agent / Env shares (% of all failures)."""

    shares: dict[str, Counter] = {}
    for model, values in counts.items():
        total = sum(values.values())
        if not total:
            shares[model] = Counter({k: 0.0 for k in SUMMARY_TYPES})
            continue
        user = sum(int(values.get(t, 0)) for t in USER_ERROR_TAXONOMY_LABELS)
        env = sum(int(values.get(t, 0)) for t in ENV_ERROR_TAXONOMY_LABELS)
        agent = total - user - env
        shares[model] = Counter(
            {
                "User Error": 100.0 * user / total,
                "Agent Error": 100.0 * agent / total,
                "Env Error": 100.0 * env / total,
            }
        )
    return shares


def series_for(counts: dict[str, Counter], base: str) -> list[str]:
    series = [m for m in sorted(counts, key=sort_key) if base_model_of(m) == base]
    if REFERENCE_MODEL in counts:
        series = [REFERENCE_MODEL] + [m for m in series if m != REFERENCE_MODEL]
    return series


def x_labels(types: list[str], *, labels: dict[str, str] | None = None) -> list[str]:
    table = labels if labels is not None else FAILURE_LABELS
    out = []
    for code in types:
        if code in table:
            out.append(table[code])
        else:
            out.append("\n".join(textwrap.wrap(code, LABEL_WIDTH)) or code)
    return out


def _register_font_weights(font_manager) -> None:
    variable = next(
        (
            f.fname
            for f in font_manager.fontManager.ttflist
            if f.name == FONT_FAMILY and f.style == "normal" and "[wght]" in f.fname
        ),
        None,
    )
    if variable is None:
        return
    try:
        from fontTools import ttLib
        from fontTools.varLib import instancer
    except ImportError:
        print(
            f"  fonttools missing; {FONT_FAMILY!r} bold/semibold will render as regular"
        )
        return

    cache = Path(__file__).resolve().parents[2] / ".fonts"
    stem = Path(variable).name.split("[")[0]
    for weight in FONT_WEIGHTS:
        face = cache / f"{stem}-{weight}.ttf"
        if not face.exists():
            cache.mkdir(parents=True, exist_ok=True)
            font = ttLib.TTFont(variable)
            instancer.instantiateVariableFont(
                font, {"wght": weight}, inplace=True, updateFontNames=True
            )
            font.save(face)
        font_manager.fontManager.addfont(str(face))


def _pyplot():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib import font_manager

    if FONT_FAMILY:
        if FONT_FAMILY in {f.name for f in font_manager.fontManager.ttflist}:
            _register_font_weights(font_manager)
            plt.rcParams["font.family"] = FONT_FAMILY
        else:
            print(f"  font {FONT_FAMILY!r} not installed; using the matplotlib default")
    plt.rcParams["hatch.linewidth"] = 0.25
    return plt


def _jagged_top_verts(
    x_center: float,
    width: float,
    height: float,
    *,
    n_teeth: int = 4,
    jag: float = 3.5,
) -> list[tuple[float, float]]:
    """Closed polygon: flat bottom, sawtooth top (peaks at ``height``).

    ``n_teeth`` must stay even so both outer edges land on a peak rather than
    leaving one side clipped mid-valley.
    """

    half = width / 2.0
    left, right = x_center - half, x_center + half
    verts: list[tuple[float, float]] = [(left, 0.0), (right, 0.0)]
    for i in range(n_teeth + 1):
        t = i / n_teeth
        x = right - t * width
        y = height - (jag if i % 2 else 0.0)
        verts.append((x, y))
    return verts


def _shade_categories(fig, column_axes, n_types: int) -> None:
    """Band every other category as one figure-level stripe.

    Drawn in figure coords rather than per-axes so a stripe runs unbroken through
    every model row (including the gaps) and down past both lines of its x label.
    """

    from matplotlib.patches import Rectangle

    # Positions and tick-label extents are only known once the figure has been laid out.
    fig.canvas.draw()
    inv = fig.transFigure.inverted()
    ax_top, ax_bottom = column_axes[0], column_axes[-1]
    top = ax_top.get_position().y1 + CATEGORY_BAND_PAD
    labels = [t for t in ax_bottom.get_xticklabels() if t.get_text()]
    if labels:
        renderer = fig.canvas.get_renderer()
        bottom = (
            min(t.get_window_extent(renderer).transformed(inv).y0 for t in labels)
            - CATEGORY_BAND_PAD
        )
    else:
        bottom = ax_bottom.get_position().y0 - CATEGORY_BAND_PAD
    for ax in column_axes:
        # Let the stripes show through; the axes' own white fill would hide them.
        ax.patch.set_visible(False)
    for i in range(1, n_types, 2):
        x0 = inv.transform(ax_top.transData.transform((i - 0.5, 0)))[0]
        x1 = inv.transform(ax_top.transData.transform((i + 0.5, 0)))[0]
        fig.add_artist(
            Rectangle(
                (x0, bottom),
                x1 - x0,
                top - bottom,
                transform=fig.transFigure,
                facecolor=CATEGORY_BAND_COLOR,
                edgecolor="none",
                zorder=0,
            )
        )


def _draw_panel(
    ax,
    counts,
    types,
    series,
    *,
    cap: float | None = None,
    bar_span: float = COMBINED_BAR_SPAN,
) -> None:
    from matplotlib.patches import Polygon

    x = np.arange(len(types))
    width = float(bar_span) / max(1, len(series))
    for i, model in enumerate(series):
        name = label_of(model)
        offset = (i - (len(series) - 1) / 2) * width
        hatch = hatch_of(model)
        color = COLORS.get(name, "#cccccc")
        raw = [counts[model].get(t, 0) for t in types]
        labeled = False
        for xi, value in zip(x + offset, raw):
            label = name if not labeled else "_nolegend_"
            if cap is not None and value > cap:
                poly = Polygon(
                    _jagged_top_verts(float(xi), width, float(cap)),
                    closed=True,
                    facecolor=color,
                    edgecolor="black",
                    linewidth=0.25,
                    hatch=hatch,
                    label=label,
                    clip_on=False,
                    zorder=3,
                )
                ax.add_patch(poly)
                # Vertical: adjacent over-cap bars are ~6pt apart, so horizontal text collides.
                ax.annotate(
                    f"{value:.0f}",
                    xy=(xi, cap),
                    xytext=(0, 1.5),
                    textcoords="offset points",
                    ha="center",
                    va="bottom",
                    rotation=90,
                    fontsize=SMALL - 3.5,
                    fontweight=600,
                    clip_on=False,
                )
            else:
                ax.bar(
                    [xi],
                    [value],
                    width,
                    label=label,
                    color=color,
                    hatch=hatch,
                    linewidth=0.25,
                    edgecolor="black",
                )
            labeled = True
    ax.set_xticks(x)
    # Snug limits: no dead margin past the outermost bars.
    ax.set_xlim(-0.5, len(types) - 0.5)
    ax.spines["right"].set_visible(False)
    ax.spines["top"].set_visible(False)


def plot_combined(
    values: dict[str, Counter],
    types: list[str],
    out_dir: Path,
    *,
    ylabel: str = "% of Agent Failures",
    ylim: tuple[float, float] = COMBINED_PERCENT_YLIM,
    stem: str = "failure_percent_combined",
    bucket_values: dict[str, Counter] | None = None,
    bucket_ylabel: str = "% of Failures",
    ytick_step: float | None = None,
    bar_span: float = COMBINED_BAR_SPAN,
    label_rotation: int = DETAIL_LABEL_ROTATION,
) -> list[Path]:
    """Per base model: optional User/Agent/Env summary (left) + agent-mode detail."""

    plt = _pyplot()
    bases = list(reversed([b for b in BASE_MODELS if series_for(values, b)]))
    if not bases:
        return []

    detail_types = agent_detail_types(types)
    buckets = bucket_values
    with_summary = buckets is not None
    n_rows = len(bases)
    n_cols = 2 if with_summary else 1
    gs_kw: dict[str, Any] = {}
    if with_summary:
        gs_kw["width_ratios"] = [SUMMARY_WIDTH_RATIO, DETAIL_WIDTH_RATIO]
        gs_kw["wspace"] = 0.18
    fig, axes = plt.subplots(
        n_rows,
        n_cols,
        figsize=COMBINED_FIGSIZE,
        sharex="col",
        gridspec_kw=gs_kw or None,
    )
    axes = np.array(axes).reshape(n_rows, n_cols)

    drawn: set[str] = set()
    for row, base in enumerate(bases):
        ax_detail = axes[row, -1]
        series = series_for(values, base)

        if with_summary:
            ax_summary = axes[row, 0]
            ax_summary.grid(
                True,
                axis="y",
                color="lightgray",
                linestyle="--",
                linewidth=0.5,
                alpha=0.5,
            )
            ax_summary.set_axisbelow(True)
            _draw_panel(
                ax_summary,
                buckets,
                list(SUMMARY_TYPES),
                series,
                cap=SUMMARY_PERCENT_YLIM[1],
                bar_span=COMBINED_BAR_SPAN,
            )
            ax_summary.tick_params(labelsize=SMALL)
            ax_summary.tick_params(axis="both", which="both", bottom=False, left=False)
            ax_summary.set_ylim(*SUMMARY_PERCENT_YLIM)
            ax_summary.set_yticks(
                np.arange(
                    SUMMARY_PERCENT_YLIM[0],
                    SUMMARY_PERCENT_YLIM[1] + 1,
                    SUMMARY_YTICK_STEP,
                )
            )
            drawn.update(ax_summary.get_legend_handles_labels()[1])

        ax_detail.grid(
            True, axis="y", color="lightgray", linestyle="--", linewidth=0.5, alpha=0.5
        )
        ax_detail.set_axisbelow(True)
        _draw_panel(
            ax_detail, values, detail_types, series, cap=ylim[1], bar_span=bar_span
        )
        ax_detail.tick_params(labelsize=SMALL)
        ax_detail.tick_params(axis="both", which="both", bottom=False, left=False)
        ax_detail.set_ylim(*ylim)
        step = COMBINED_YTICK_STEP if ytick_step is None else ytick_step
        ax_detail.set_yticks(np.arange(ylim[0], ylim[1] + 1, step))
        drawn.update(ax_detail.get_legend_handles_labels()[1])

    # Upright labels center under their band; tilted ones anchor at their right edge so
    # the stacked lines don't fan out into the next category.
    tilted = bool(label_rotation)
    label_kw: dict[str, Any] = (
        {
            "rotation": label_rotation,
            "ha": "right",
            "rotation_mode": "anchor",
            "multialignment": "right",
        }
        if tilted
        else {"rotation": 0, "ha": "center", "multialignment": "center"}
    )
    if with_summary:
        axes[-1, 0].set_xticklabels(
            x_labels(list(SUMMARY_TYPES), labels=SUMMARY_LABELS),
            fontsize=SMALL,
            **label_kw,
        )
    ax_x = axes[-1, -1]
    ax_x.tick_params(axis="x", which="both", bottom=False, labelbottom=True)
    ax_x.set_xticks(np.arange(len(detail_types)))
    ax_x.set_xticklabels(
        x_labels(detail_types),
        # Upright labels span the full slot width; two-line names keep neighbours from colliding.
        fontsize=SMALL if tilted else SMALL - 1,
        linespacing=0.7 if tilted else 1.0,
        **label_kw,
    )
    mid = n_rows // 2
    if with_summary:
        axes[mid, 0].set_ylabel(bucket_ylabel, fontsize=MEDIUM, fontweight=600)
    axes[mid, -1].set_ylabel(ylabel, fontsize=MEDIUM, fontweight=600)

    from matplotlib.patches import Patch

    ordered = list(dict.fromkeys(n for n in MODEL_LABELS.values() if n in drawn))
    fig.legend(
        [
            Patch(facecolor=COLORS[n], edgecolor="black", linewidth=0.25)
            for n in ordered
        ],
        ordered,
        loc="upper center",
        ncol=len(ordered),
        fontsize=SMALL,
        frameon=False,
        bbox_to_anchor=(0.5, 1.05),
        columnspacing=1.2,
        handlelength=1.2,
    )
    fig.legend(
        handles=[
            Patch(
                facecolor="white",
                edgecolor="black",
                hatch=h,
                linewidth=0.25,
                label=text,
            )
            for text, h in CONDITIONING_LEGEND
        ],
        loc="upper center",
        bbox_to_anchor=(0.5, 1.0),
        ncol=2,
        fontsize=SMALL,
        frameon=False,
        handlelength=1.2,
        columnspacing=1.0,
    )
    # Extra inter-row gap so per-row model titles sit clearly above the panels.
    fig.subplots_adjust(
        left=0.08, right=0.99, top=0.80, bottom=0.22, wspace=0.28, hspace=0.75
    )
    for row, base in enumerate(bases):
        bbox0 = axes[row, 0].get_position()
        bbox1 = axes[row, -1].get_position()
        fig.text(
            (bbox0.x0 + bbox1.x1) / 2.0 - (0.025 if with_summary else 0.0),
            bbox0.y1 + 0.045,
            MODEL_NAMES[base],
            ha="center",
            va="center",
            fontsize=MEDIUM + 1,
            fontweight=600,
        )

    _shade_categories(fig, list(axes[:, -1]), len(detail_types))
    if with_summary:
        _shade_categories(fig, list(axes[:, 0]), len(SUMMARY_TYPES))

    out_dir.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []
    for suffix in ("pdf", "png"):
        path = out_dir / f"{stem}.{suffix}"
        fig.savefig(path, dpi=DPI, bbox_inches="tight", pad_inches=0.12)
        written.append(path)
    plt.close(fig)
    return written


def plot_combined_percent(
    counts: dict[str, Counter],
    types: list[str],
    out_dir: Path,
    *,
    stem: str = "failure_percent_combined",
) -> list[Path]:
    detail_types = agent_detail_types(types)
    agent_values = as_agent_percent(counts, detail_types)
    written = plot_combined(
        agent_values,
        types,
        out_dir,
        ylabel="% of Agent Failures",
        ylim=COMBINED_PERCENT_YLIM,
        stem=stem,
        bucket_values=as_bucket_percent(counts),
        bucket_ylabel="% of Failures",
    )
    agent_stem = stem.replace("failure_percent_combined", "failure_percent_agent", 1)
    if agent_stem == stem:
        agent_stem = f"{stem}_agent"
    written.extend(
        plot_combined(
            agent_values,
            types,
            out_dir,
            ylabel="% of Agent Failures",
            ylim=AGENT_PERCENT_YLIM,
            ytick_step=AGENT_YTICK_STEP,
            bar_span=AGENT_BAR_SPAN,
            label_rotation=AGENT_LABEL_ROTATION,
            stem=agent_stem,
        )
    )
    return written


def write_matrix(path: Path, per_model: dict[str, Counter], types: list[str]) -> None:
    import csv

    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow(["base_model", "source_id", "family", "arm", "n"] + types)
        for model in sorted(per_model, key=sort_key):
            family, _sim, arm = parse_source(model)
            values = per_model[model]
            writer.writerow(
                [
                    base_model_of(model),
                    model,
                    family,
                    arm or "",
                    sum(values.values()),
                    *[values.get(t, 0) for t in types],
                ]
            )


def run_plot(
    run_dir: Path,
    *,
    out_dir: Path | None = None,
    exclude_labels: Iterable[str] | None = None,
    merge_labels: dict[str, str] | None = None,
) -> dict[str, Any]:
    drop = {str(x).strip() for x in (exclude_labels or []) if str(x).strip()}
    merge = {
        str(k).strip(): str(v).strip()
        for k, v in (merge_labels or {}).items()
        if str(k).strip() and str(v).strip() and str(k).strip() != str(v).strip()
    }
    # Default paper cut (unless caller already remapped): Early Stop → Env;
    # Identity / Leakage / legacy Critical → Other Sim (plotted as User Error).
    for src, dst in PAPER_CUT_MERGE.items():
        if src not in merge:
            merge[src] = dst
    # Uncategorized is a residual bucket, not a failure mode for the paper figure.
    drop.add("Uncategorized")
    counts, types = load_counts(run_dir, exclude_labels=drop, merge_labels=merge)
    types = order_types_by_human(counts, types)
    dest = out_dir or (run_dir / "figures")
    dest.mkdir(parents=True, exist_ok=True)
    from cue_training.evaluation.fma.sim_errors import SIMULATOR_ERROR_LABELS

    suffix = (
        "_no_sim_error"
        if set(SIMULATOR_ERROR_LABELS).issubset(drop)
        or "Critical User Simulator Error" in drop
        else ("_excl" if drop - {"Uncategorized"} else "")
    )
    paper_cut = all(merge.get(src) == dst for src, dst in PAPER_CUT_MERGE.items())
    user_error_bucket = bool(merge) and all(
        merge.get(src) == USER_ERROR_BUCKET_TARGET for src in USER_ERROR_BUCKET_SOURCES
    )
    # Paper cut / --user-error keep stem failure_percent_combined.
    if (
        merge
        and "User Data Leakage" in merge
        and not paper_cut
        and not user_error_bucket
    ):
        suffix = f"{suffix}_leakage_as_sim" if suffix else "_leakage_as_sim"
    stem = f"failure_percent_combined{suffix}"
    write_matrix(dest / f"failure_type_counts{suffix}.csv", counts, types)
    write_matrix(
        dest / f"failure_type_percentages{suffix}.csv", as_percent(counts), types
    )
    written = plot_combined_percent(counts, types, dest, stem=stem)
    return {
        "n_sources": len(counts),
        "n_types": len(types),
        "exclude_labels": sorted(drop),
        "merge_labels": merge,
        "out_dir": str(dest),
        "figures": [str(p) for p in written],
        "sources": sorted(counts, key=sort_key),
        "types": types,
    }
