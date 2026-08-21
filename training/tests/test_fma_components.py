"""Tests for fidelity/calibration component averaging, group filters, series pick."""

from __future__ import annotations

from cue_training.evaluation.fma.compare_fidelity_calibration import Row
from cue_training.evaluation.fma.plots.plot_failure_types import (
    PAIRED_SPECS,
    SAMPLED_SPECS,
)
from cue_training.evaluation.fma.plots.plot_fidelity_components import (
    component_points,
    component_scores,
    in_group,
    is_paired_row,
    scale_fidelity,
    select_series,
    series_label,
)


def _row(**kwargs) -> Row:
    defaults = {
        "source_id": "x",
        "family": "cue-general",
        "simulator": "llama",
        "arm": "paired",
        "fidelity": {},
        "abs_delta_success": 0.1,
        "user_error_pct": 20.0,
        "tvd_agent": 0.3,
        "success": 0.5,
        "delta_success": -0.1,
    }
    defaults.update(kwargs)
    return Row(**defaults)


def test_naturalness_averages_s2r_and_inverted_turing() -> None:
    # TT=0.25 → 1 − 2*0.25 = 0.5; mean with S2R 0.7 is 0.6
    scores = component_scores(_row(fidelity={"Nat-S2R": 0.7, "Nat-TT": 0.25}))
    assert scores["Naturalness"] == 0.6
    assert scale_fidelity("Nat-TT", 0.0) == 1.0
    assert scale_fidelity("Nat-TT", 0.5) == 0.0


def test_mimicry_averages_ava_and_pt3() -> None:
    scores = component_scores(_row(fidelity={"Mim-AVA": 50.0, "Mim-PT3": 0.25}))
    assert scores["User Mimicry"] == 0.375


def test_coverage_and_calibration_scales() -> None:
    scores = component_scores(
        _row(
            fidelity={"Cov-S2RChamfer": 0.4, "Cov-SDChamfer": 0.6},
            abs_delta_success=0.2,
            user_error_pct=40.0,
            tvd_agent=0.15,
        )
    )
    assert scores["Population Coverage"] == 0.5
    assert scores["Role Adherence"] == 0.6
    assert scores["Failure Mode Reproduction"] == 0.85
    assert scores["Success Rate Alignment"] == 0.8


def test_partial_component_uses_available_metric() -> None:
    scores = component_scores(_row(fidelity={"Nat-S2R": 0.4}))
    assert scores["Naturalness"] == 0.4
    assert scores["User Mimicry"] is None


def test_fma_group_filters_match_bar_plot() -> None:
    assert in_group("cue-general", "paired", PAIRED_SPECS)
    assert in_group("ppol", "as_is", PAIRED_SPECS)
    assert in_group("realusersim_paired_noex", "paired", PAIRED_SPECS)
    assert not in_group("realusersim", "as_is", PAIRED_SPECS)
    assert in_group("realusersim", "as_is", SAMPLED_SPECS)
    assert in_group("userlm", "as_is", SAMPLED_SPECS)
    assert in_group("usp", "sample_diverse", SAMPLED_SPECS)
    assert not in_group("human", "human", PAIRED_SPECS)
    assert is_paired_row("cue-general", "paired")
    assert not is_paired_row("cue-general", "sampled_uncond")


def test_series_falls_back_to_llama_only_families() -> None:
    rows = [
        _row(family="cue-general", simulator="gemini", arm="paired"),
        _row(family="cue-general", simulator="llama", arm="paired"),
        _row(family="usp", simulator="llama", arm="paired"),
        _row(family="userlm", simulator="llama", arm="as_is"),
        _row(family="cue-refined", simulator="gemini", arm="paired"),
    ]
    picked = select_series(rows, simulator="gemini", fallback="llama")
    assert [(r.family, r.simulator) for r in picked] == [
        ("userlm", "llama"),
        ("usp", "llama"),
        ("cue-general", "gemini"),
    ]
    assert series_label(picked[0], simulator="gemini") == "UserLM (Llama)"
    assert series_label(picked[2], simulator="gemini") == "CUE"


def test_naturalness_takes_max_mode_and_coverage_stays_sampled() -> None:
    paired = _row(
        family="cue-general",
        arm="paired",
        fidelity={"Nat-S2R": 0.4, "Mim-PT3": 0.3, "Cov-S2RChamfer": 0.1},
    )
    sampled = _row(
        family="cue-general",
        arm="sampled_uncond",
        fidelity={"Nat-S2R": 0.5, "Cov-S2RChamfer": 0.6},
    )
    ppol = _row(
        family="ppol",
        arm="as_is",
        fidelity={"Nat-S2R": 0.9, "Mim-PT3": 0.2, "Cov-S2RChamfer": 0.8},
    )
    series = select_series([ppol, paired, sampled], simulator="llama")
    points = component_points(series)
    arms = {(key, row.family): row.arm for key, row, _ in points}
    scores = {(key, row.family): value for key, row, value in points}
    # Sampled Nat-S2R 0.5 beats paired 0.4; PPol only has as_is so that wins.
    assert arms[("Naturalness", "cue-general")] == "sampled_uncond"
    assert scores[("Naturalness", "ppol")] == 0.9
    assert arms[("Naturalness", "ppol")] == "as_is"
    assert arms[("Population Coverage", "cue-general")] == "sampled_uncond"
    assert arms[("User Mimicry", "cue-general")] == "paired"
    assert arms[("User Mimicry", "ppol")] == "as_is"
    assert ("Population Coverage", "ppol") not in arms
