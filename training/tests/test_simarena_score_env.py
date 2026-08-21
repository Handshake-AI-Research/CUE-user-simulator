"""Tests for SimulatorArena env scoring (math correctness + writing ratings)."""

from __future__ import annotations

import json
from pathlib import Path

from cue_training.evaluation.simulatorarena.score_env import (
    _file_name,
    discover_jobs,
    to_paper_math,
    to_paper_writing,
)


def test_to_paper_math_fills_problem_and_messages() -> None:
    native = {
        "gpt-4o": {
            "12": {
                "w1": {
                    "conversation": [["user", "help"], ["assistant", "try 4"]],
                    "user_queries": ["help"],
                    "ai_responses": ["try 4"],
                }
            }
        }
    }
    paper = to_paper_math(native, {"12": "What is 2+2?"})
    rec = paper["gpt-4o"]["12"]["w1"]
    assert rec["problem"] == "What is 2+2?"
    assert rec["first_query_content"] == "help"
    assert rec["assistant_messages"][0] == {"role": "user", "content": "help"}


def test_to_paper_writing_nests_doc_type() -> None:
    native = {
        "gpt-4o": {
            "Blog Post": {
                "inform": {
                    "w1": {
                        "conversation": [
                            ["user", "write a blog"],
                            ["assistant", "Here is a long enough draft of the blog post."],
                        ]
                    }
                }
            }
        }
    }
    paper = to_paper_writing(native)
    rec = paper["gpt-4o"]["Blog Post"]["inform"]["w1"]
    assert rec["assistant_messages"][1]["role"] == "assistant"


def test_discover_jobs_from_jsonl(tmp_path: Path) -> None:
    job = (
        tmp_path
        / "seed-0"
        / "ppol"
        / "simulatorarena"
        / "math"
        / "baseline"
        / "ppol"
        / "llama"
    )
    job.mkdir(parents=True)
    (job / "rollout.simulatorarena.jsonl").write_text(
        json.dumps(
            {
                "rollout_conversation": [
                    {"role": "user", "content": "hi"},
                    {"role": "assistant", "content": "hello"},
                ],
                "metadata": {
                    "variant": "ppol",
                    "arm": "as_is",
                    "assistant_model": "gpt-4o",
                    "workerId": "w1",
                    "problem_id": "3",
                },
            }
        )
        + "\n",
        encoding="utf-8",
    )
    jobs = discover_jobs(tmp_path, domains={"math"})
    assert len(jobs) == 1
    assert jobs[0].domain == "math"
    assert jobs[0].native_path.name == "ppol__as_is.json"
    native = json.loads(jobs[0].native_path.read_text(encoding="utf-8"))
    assert "gpt-4o" in native


def test_file_name_is_flat_slug(tmp_path: Path) -> None:
    job = tmp_path / "seed-0" / "cue" / "llama"
    job.mkdir(parents=True)
    name = _file_name(tmp_path, job, "decoder__paired")
    assert "/" not in name
    assert name.startswith("seed-0__cue__llama__decoder__paired")
