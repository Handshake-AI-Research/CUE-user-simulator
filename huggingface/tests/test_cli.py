"""The cue-hf CLI: generate from sessions, sample synthetic users."""

from __future__ import annotations

import json

import pytest
import torch
from cue_hf.cli import main
from tests.conftest import make_sampler_payload


@pytest.fixture
def checkpoint(tiny_model, tmp_path):
    tiny_model.save_pretrained(tmp_path)
    torch.save(make_sampler_payload(tiny_model.config.bottleneck_dim), tmp_path / "sampler.pt")
    return tmp_path


def test_generate_writes_one_manual_per_session(checkpoint, tmp_path, capsys):
    session = tmp_path / "session.json"
    session.write_text(
        json.dumps(
            {
                "id": "s0",
                "turns": [
                    {"role": "user", "content": "hey fix this"},
                    {"role": "assistant", "content": "Sure."},
                    {"role": "user", "content": "shorter"},
                ],
            }
        ),
        encoding="utf-8",
    )
    assert main(["generate", "--checkpoint", str(checkpoint), "--input", str(session)]) == 0
    rows = [json.loads(line) for line in capsys.readouterr().out.splitlines()]
    assert len(rows) == 1
    assert rows[0]["render"] is None


def test_generate_reads_jsonl_batches_and_renders(checkpoint, tmp_path, capsys):
    sessions = tmp_path / "sessions.jsonl"
    sessions.write_text(
        "\n".join(json.dumps({"id": f"s{i}", "turns": [{"role": "user", "content": f"hi {i}"}]}) for i in range(2)),
        encoding="utf-8",
    )
    out = tmp_path / "out.jsonl"
    assert (
        main(
            [
                "generate",
                "--checkpoint",
                str(checkpoint),
                "--input",
                str(sessions),
                "--render",
                "--output",
                str(out),
            ]
        )
        == 0
    )
    rows = [json.loads(line) for line in out.read_text(encoding="utf-8").splitlines()]
    assert len(rows) == 2
    assert all(isinstance(row["render"], str) for row in rows)


def test_sample_emits_n_manuals(checkpoint, capsys):
    assert main(["sample", "--checkpoint", str(checkpoint), "--n", "3", "--seed", "1"]) == 0
    assert len(capsys.readouterr().out.strip().splitlines()) == 3
