"""Tests for τ² pairwise-success composition bars."""

from __future__ import annotations

import json
from pathlib import Path

from cue_training.evaluation.fma.plots.plot_pairwise_success import (
    as_percent,
    keep_series,
    load_confusion,
)


def _write_summary(
    path: Path, *, arm: str, both_ok: int, both_fail: int, h_only: int, s_only: int
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    group = f"{arm}/airline"
    path.write_text(
        json.dumps(
            {
                "aggregates_by_arm": {
                    group: [
                        {
                            "metric_name": "env/tau2_task_success",
                            "extras": {
                                "success_confusion_vs_human": {
                                    "both_success": both_ok,
                                    "human_success_rollout_fail": h_only,
                                    "human_fail_rollout_success": s_only,
                                    "both_fail": both_fail,
                                }
                            },
                        }
                    ]
                },
                "group_meta": {group: {"arm": arm, "domain": "airline"}},
            }
        ),
        encoding="utf-8",
    )


def test_load_sums_seeds_and_filters_fma_groups(tmp_path: Path) -> None:
    root = tmp_path / "_runs" / "customer-service"
    _write_summary(
        root / "seed-0" / "cue-general" / "llama" / "summary.json",
        arm="paired",
        both_ok=10,
        both_fail=20,
        h_only=5,
        s_only=5,
    )
    _write_summary(
        root / "seed-1" / "cue-general" / "llama" / "summary.json",
        arm="paired",
        both_ok=10,
        both_fail=20,
        h_only=5,
        s_only=5,
    )
    _write_summary(
        root / "seed-0" / "base" / "llama" / "summary.json",
        arm="as_is",
        both_ok=99,
        both_fail=1,
        h_only=0,
        s_only=0,
    )
    counts = load_confusion(root)
    sid = "cue-general-llama|paired"
    assert sid in counts
    assert "base-llama|as_is" not in counts
    assert counts[sid]["both_success"] == 20
    assert counts[sid]["both_fail"] == 40
    pct = as_percent(counts)[sid]
    assert pct["agree"] == 75.0
    assert pct["disagree"] == 25.0
    assert abs(sum(pct.values()) - 100.0) < 1e-9


def test_keep_series_gemini_paired_only() -> None:
    assert keep_series("cue-general-gemini|paired")
    assert keep_series("ppol-gemini|as_is")
    assert keep_series("realusersim_paired_noex-gemini|paired")
    assert not keep_series("cue-general-gemini|sampled_uncond")
    assert not keep_series("realusersim-gemini|as_is")
    assert not keep_series("cue-general-llama|paired")
