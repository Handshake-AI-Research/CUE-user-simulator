"""Baseline paper-parity manifest validation tests."""

from __future__ import annotations

from pathlib import Path

import pytest

from cue_training.baselines.common.manifest import (
    IncompleteArtifactError,
    PAPER_MANIFESTS,
    train_paper,
    validate_artifacts,
    write_training_manifest,
)


@pytest.mark.parametrize("method", sorted(PAPER_MANIFESTS))
def test_paper_manifest_exists(method: str):
    paper = PAPER_MANIFESTS[method]
    assert paper.preset
    assert paper.stages
    # train_paper reports `preset`, so a lean stand-in must never be the headline preset.
    assert paper.preset != paper.lean_preset


def test_validate_requires_complete_manifest(tmp_path: Path):
    art = tmp_path / "realusersim"
    art.mkdir()
    (art / "profiles.jsonl").write_text("{}\n", encoding="utf-8")
    with pytest.raises(IncompleteArtifactError):
        validate_artifacts("realusersim", art, require_complete=True)


def test_validate_accepts_complete_realusersim(tmp_path: Path):
    art = tmp_path / "realusersim"
    art.mkdir()
    (art / "profiles.jsonl").write_text("{}\n", encoding="utf-8")
    write_training_manifest(
        art,
        paper=PAPER_MANIFESTS["realusersim"],
        status="complete",
        stages_done=["extract_profiles"],
    )
    out = validate_artifacts("realusersim", art, require_complete=True)
    assert out["status"] == "complete"


def test_train_paper_userlm_dry_run(tmp_path: Path):
    out = train_paper(
        "userlm",
        artifact_root=tmp_path,
        data_path=None,
        dry_run=True,
    )
    assert (out / "manifest.json").is_file()
