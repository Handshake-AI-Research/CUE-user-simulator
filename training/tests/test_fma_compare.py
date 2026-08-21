"""Tests for fidelity × calibration comparison table."""

from __future__ import annotations

from pathlib import Path

import pytest
from cue_training.evaluation.fma.compare_fidelity_calibration import (
    FIDELITY,
    correlations,
    fit_rows_for,
    format_corr_markdown,
    join_rows,
    run_compare,
)


def test_join_and_correlations_synthetic() -> None:
    # Three methods with aligned fidelity / calibration so r is defined.
    lb = {
        ("human", "", "human"): {"env/tau2_success_rate": 0.80},
        ("ppol", "llama", "as_is"): {
            "env/tau2_success_rate": 0.70,
            "classifier/sim2real": 0.1,
            "judge/turing_sonnet_qwen": 0.4,
            "mimicry/wegmann_ava": 10.0,
            "mimicry/paired_audit": 0.1,
            "coverage/sim2real_behavioral": 0.1,
            "coverage/styledistance_behavioral": 0.1,
        },
        ("ppol", "gpt", "as_is"): {
            "env/tau2_success_rate": 0.60,
            "classifier/sim2real": 0.2,
            "judge/turing_sonnet_qwen": 0.3,
            "mimicry/wegmann_ava": 20.0,
            "mimicry/paired_audit": 0.2,
            "coverage/sim2real_behavioral": 0.2,
            "coverage/styledistance_behavioral": 0.2,
        },
        ("ppol", "gemini", "as_is"): {
            "env/tau2_success_rate": 0.50,
            "classifier/sim2real": 0.3,
            "judge/turing_sonnet_qwen": 0.2,
            "mimicry/wegmann_ava": 30.0,
            "mimicry/paired_audit": 0.3,
            "coverage/sim2real_behavioral": 0.3,
            "coverage/styledistance_behavioral": 0.3,
        },
    }
    cal = {
        "ppol-llama|as_is": {"sim_error_rate": 0.10, "tvd_agent": 0.10},
        "ppol-gpt|as_is": {"sim_error_rate": 0.20, "tvd_agent": 0.20},
        "ppol-gemini|as_is": {"sim_error_rate": 0.30, "tvd_agent": 0.30},
    }
    rows, notes = join_rows(lb, cal)
    assert notes == []
    assert len(rows) == 3
    by_sid = {r.source_id: r for r in rows}
    assert by_sid["ppol-llama|as_is"].abs_delta_success == pytest.approx(0.10)
    assert by_sid["ppol-llama|as_is"].user_error_pct == pytest.approx(10.0)
    corr, ns = correlations(rows)
    # S2R rises with |ΔSuc|, user error, and TVD → r ≈ 1
    assert corr["Nat-S2R_vs_abs_delta_success"] == 1.0
    assert corr["Nat-S2R_vs_user_error_pct"] == 1.0
    assert corr["Nat-S2R_vs_tvd_agent"] == 1.0
    assert ns["Nat-S2R_vs_abs_delta_success"] == 3
    # PPol as_is is paired → coverage cells empty / undefined
    assert "Cov-S2RChamfer_vs_abs_delta_success" not in corr
    assert ns["Cov-S2RChamfer_vs_abs_delta_success"] == 0
    md = format_corr_markdown(corr, ns)
    assert "Nat-S2R" in md and "TVD_Agent" in md
    assert all(short in md for _k, short in FIDELITY)


def test_pairwise_keeps_row_missing_one_metric() -> None:
    lb = {
        ("human", "", "human"): {"env/tau2_success_rate": 0.80},
        ("userlm", "llama", "as_is"): {
            "env/tau2_success_rate": 0.70,
            "classifier/sim2real": 0.1,
            "judge/turing_sonnet_qwen": 0.4,
            # no AVA
            "mimicry/paired_audit": 0.1,
            "coverage/sim2real_behavioral": 0.1,
            "coverage/styledistance_behavioral": 0.1,
        },
        ("userlm", "gpt", "as_is"): {
            "env/tau2_success_rate": 0.60,
            "classifier/sim2real": 0.2,
            "judge/turing_sonnet_qwen": 0.3,
            "mimicry/wegmann_ava": 20.0,
            "mimicry/paired_audit": 0.2,
            "coverage/sim2real_behavioral": 0.2,
            "coverage/styledistance_behavioral": 0.2,
        },
        ("userlm", "gemini", "as_is"): {
            "env/tau2_success_rate": 0.50,
            "classifier/sim2real": 0.3,
            "judge/turing_sonnet_qwen": 0.2,
            "mimicry/wegmann_ava": 30.0,
            "mimicry/paired_audit": 0.3,
            "coverage/sim2real_behavioral": 0.3,
            "coverage/styledistance_behavioral": 0.3,
        },
    }
    cal = {
        "userlm-llama|as_is": {"sim_error_rate": 0.10, "tvd_agent": 0.10},
        "userlm-gpt|as_is": {"sim_error_rate": 0.20, "tvd_agent": 0.20},
        "userlm-gemini|as_is": {"sim_error_rate": 0.30, "tvd_agent": 0.30},
    }
    rows, notes = join_rows(lb, cal)
    assert len(rows) == 3
    assert any("mimicry/wegmann_ava" in n for n in notes)
    assert (
        "Mim-AVA" not in rows[0].fidelity or rows[0].source_id != "userlm-llama|as_is"
    )
    llama = next(r for r in rows if r.source_id == "userlm-llama|as_is")
    assert "Mim-AVA" not in llama.fidelity
    assert "Nat-S2R" in llama.fidelity
    _corr, ns = correlations(rows)
    assert ns["Nat-S2R_vs_abs_delta_success"] == 3
    assert ns["Mim-AVA_vs_abs_delta_success"] == 2
    # userlm as_is is sampled → coverage includes all three
    assert ns["Cov-S2RChamfer_vs_abs_delta_success"] == 3
    assert len(fit_rows_for(rows, "Cov-S2RChamfer")) == 3


def test_run_compare_writes_artifacts(tmp_path: Path) -> None:
    import json

    lb_path = tmp_path / "leaderboard.json"
    dist_path = tmp_path / "distributions.json"
    out_dir = tmp_path / "out"

    lb_path.write_text(
        json.dumps(
            {
                "rows": [
                    {
                        "domain": "macro",
                        "family": "human",
                        "simulator": "—",
                        "arm": "human",
                        "metrics": {
                            "env/tau2_success_rate": {"mean": 0.8},
                        },
                    },
                    {
                        "domain": "macro",
                        "family": "userlm",
                        "simulator": "llama",
                        "arm": "as_is",
                        "metrics": {
                            "env/tau2_success_rate": {"mean": 0.7},
                            "classifier/sim2real": {"mean": 0.1},
                            "judge/turing_sonnet_qwen": {"mean": 0.4},
                            "mimicry/wegmann_ava": {"mean": 10.0},
                            "mimicry/paired_audit": {"mean": 0.1},
                            "coverage/sim2real_behavioral": {"mean": 0.1},
                            "coverage/styledistance_behavioral": {"mean": 0.1},
                        },
                    },
                    {
                        "domain": "macro",
                        "family": "userlm",
                        "simulator": "gpt",
                        "arm": "as_is",
                        "metrics": {
                            "env/tau2_success_rate": {"mean": 0.6},
                            "classifier/sim2real": {"mean": 0.2},
                            "judge/turing_sonnet_qwen": {"mean": 0.3},
                            "mimicry/wegmann_ava": {"mean": 20.0},
                            "mimicry/paired_audit": {"mean": 0.2},
                            "coverage/sim2real_behavioral": {"mean": 0.2},
                            "coverage/styledistance_behavioral": {"mean": 0.2},
                        },
                    },
                    {
                        "domain": "macro",
                        "family": "userlm",
                        "simulator": "gemini",
                        "arm": "as_is",
                        "metrics": {
                            "env/tau2_success_rate": {"mean": 0.5},
                            "classifier/sim2real": {"mean": 0.3},
                            "judge/turing_sonnet_qwen": {"mean": 0.2},
                            "mimicry/wegmann_ava": {"mean": 30.0},
                            "mimicry/paired_audit": {"mean": 0.3},
                            "coverage/sim2real_behavioral": {"mean": 0.3},
                            "coverage/styledistance_behavioral": {"mean": 0.3},
                        },
                    },
                ]
            }
        ),
        encoding="utf-8",
    )
    dist_path.write_text(
        json.dumps(
            {
                "comparisons": {
                    "userlm-llama|as_is": {"sim_error_rate": 0.1, "tvd_agent": 0.1},
                    "userlm-gpt|as_is": {"sim_error_rate": 0.2, "tvd_agent": 0.2},
                    "userlm-gemini|as_is": {"sim_error_rate": 0.3, "tvd_agent": 0.3},
                }
            }
        ),
        encoding="utf-8",
    )

    result = run_compare(
        leaderboard=lb_path,
        distributions=dist_path,
        out_dir=out_dir,
        plot=False,
    )
    assert result["n_rows"] == 3
    assert Path(result["points_csv"]).is_file()
    assert Path(result["correlations_csv"]).is_file()
    assert Path(result["correlations_md"]).is_file()
