"""Unit tests for FMA percent-combined plot helpers."""

from __future__ import annotations

from pathlib import Path

from cue_training.evaluation.fma.plots.plot_failure_types import (
    PAIRED_HATCH,
    SAMPLED_HATCH,
    as_percent,
    hatch_of,
    is_included,
    label_of,
    load_counts,
    order_types_by_human,
    parse_source,
)
from cue_training.evaluation.fma.store import save_taxonomy, write_jsonl


def test_parse_and_hatch_switched_arms() -> None:
    assert parse_source("ppol-llama|as_is") == ("ppol", "llama", "as_is")
    assert parse_source("cue-general-gpt|sampled_uncond") == (
        "cue-general",
        "gpt",
        "sampled_uncond",
    )
    assert parse_source("human") == ("human", None, None)

    # Switched vs old FMA defaults: PPol as_is is paired; RealUserSim as_is is sampled.
    assert hatch_of("ppol-llama|as_is") == PAIRED_HATCH
    assert hatch_of("usp-llama|paired") == PAIRED_HATCH
    assert hatch_of("cue-general-llama|paired") == PAIRED_HATCH
    assert hatch_of("realusersim_paired_noex-gpt|paired") == PAIRED_HATCH
    assert hatch_of("realusersim-llama|as_is") == SAMPLED_HATCH
    assert hatch_of("usp-llama|sample_diverse") == SAMPLED_HATCH
    assert hatch_of("cue-general-llama|sampled_uncond") == SAMPLED_HATCH
    assert hatch_of("userlm-llama|as_is") == SAMPLED_HATCH
    assert hatch_of("human") == ""


def test_include_filter_skips_base() -> None:
    assert is_included("human")
    assert is_included("ppol-gpt|as_is")
    assert is_included("usp-llama|sample_diverse")
    assert is_included("realusersim_paired_noex-gemini|paired")
    assert not is_included("base-llama|as_is")
    assert not is_included("ppol-llama|paired")  # wrong arm for this paper split
    assert label_of("cue-general-gemini|paired") == "CUE"
    # Paired-noex shares the RealUserSim family label with sampled as_is.
    assert label_of("realusersim_paired_noex-gpt|paired") == "RealUserSim"
    assert label_of("realusersim-gpt|as_is") == "RealUserSim"


def test_as_percent_and_load_counts(tmp_path: Path) -> None:
    save_taxonomy(
        tmp_path,
        [
            {"name": "Wrong Action Parameter", "description": "", "examples": []},
            {"name": "Unnecessary Escalation", "description": "", "examples": []},
        ],
    )
    write_jsonl(
        tmp_path / "tagged.jsonl",
        [
            {
                "primary_key": "h1",
                "source_id": "human",
                "label": "Wrong Action Parameter",
            },
            {
                "primary_key": "h2",
                "source_id": "human",
                "label": "Unnecessary Escalation",
            },
            {
                "primary_key": "p1",
                "source_id": "ppol-llama|as_is",
                "label": "Wrong Action Parameter",
            },
            {
                "primary_key": "p2",
                "source_id": "ppol-llama|as_is",
                "label": "Wrong Action Parameter",
            },
            {
                "primary_key": "b1",
                "source_id": "base-llama|as_is",
                "label": "Unnecessary Escalation",
            },
        ],
    )
    counts, types = load_counts(tmp_path)
    assert "base-llama|as_is" not in counts
    assert counts["ppol-llama|as_is"]["Wrong Action Parameter"] == 2
    assert "Wrong Action Parameter" in types
    shares = as_percent(counts)
    assert shares["ppol-llama|as_is"]["Wrong Action Parameter"] == 100.0
    assert shares["human"]["Wrong Action Parameter"] == 50.0


def test_load_counts_excludes_sim_error(tmp_path: Path) -> None:
    save_taxonomy(
        tmp_path,
        [
            {"name": "Wrong Action Parameter", "description": "", "examples": []},
            {
                "name": "Critical User Simulator Error",
                "description": "",
                "examples": [],
            },
        ],
    )
    write_jsonl(
        tmp_path / "tagged.jsonl",
        [
            {
                "primary_key": "h1",
                "source_id": "human",
                "label": "Wrong Action Parameter",
            },
            {
                "primary_key": "h2",
                "source_id": "human",
                "label": "Critical User Simulator Error",
            },
            {
                "primary_key": "p1",
                "source_id": "ppol-llama|as_is",
                "label": "Critical User Simulator Error",
            },
            {
                "primary_key": "p2",
                "source_id": "ppol-llama|as_is",
                "label": "Wrong Action Parameter",
            },
        ],
    )
    counts, types = load_counts(
        tmp_path, exclude_labels=["Critical User Simulator Error"]
    )
    assert "Critical User Simulator Error" not in types
    assert counts["human"]["Wrong Action Parameter"] == 1
    assert sum(counts["human"].values()) == 1
    assert counts["ppol-llama|as_is"]["Wrong Action Parameter"] == 1
    assert sum(counts["ppol-llama|as_is"].values()) == 1


def test_order_types_by_human_and_omit_uncategorized(tmp_path: Path) -> None:
    save_taxonomy(
        tmp_path,
        [
            {"name": "Wrong Action Parameter", "description": "", "examples": []},
            {"name": "Unnecessary Escalation", "description": "", "examples": []},
            {"name": "Uncategorized", "description": "", "examples": []},
        ],
    )
    write_jsonl(
        tmp_path / "tagged.jsonl",
        [
            {
                "primary_key": "h1",
                "source_id": "human",
                "label": "Unnecessary Escalation",
            },
            {
                "primary_key": "h2",
                "source_id": "human",
                "label": "Unnecessary Escalation",
            },
            {
                "primary_key": "h3",
                "source_id": "human",
                "label": "Wrong Action Parameter",
            },
            {"primary_key": "h4", "source_id": "human", "label": "Uncategorized"},
            {
                "primary_key": "p1",
                "source_id": "ppol-llama|as_is",
                "label": "Wrong Action Parameter",
            },
        ],
    )
    counts, types = load_counts(tmp_path, exclude_labels=["Uncategorized"])
    assert "Uncategorized" not in types
    assert order_types_by_human(counts, types) == [
        "Unnecessary Escalation",
        "Wrong Action Parameter",
    ]
