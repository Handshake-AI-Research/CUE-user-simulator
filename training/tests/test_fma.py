"""Tests for lightweight FMA ingest → propose (stub) → decisions → tag → TVD."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from cue_training.evaluation.common.io import read_jsonl
from cue_training.evaluation.fma.conversation import normalize_conversation
from cue_training.evaluation.fma.criteria import load_task_criteria
from cue_training.evaluation.fma.ingest import ingest_human, ingest_sim
from cue_training.evaluation.fma.propose import propose_batch
from cue_training.evaluation.fma.review import (
    finish_batch,
    merge_failure_modes,
    rename_failure_mode,
)
from cue_training.evaluation.fma.sample import sample_batch, sample_uncategorized_batch
from cue_training.evaluation.fma.store import load_taxonomy, save_taxonomy, write_jsonl
from cue_training.evaluation.fma.tag import tag_remainder
from cue_training.evaluation.fma.tvd import compute_distributions, total_variation


def test_normalize_inline_and_structured_tools() -> None:
    structured = normalize_conversation(
        [
            {"role": "user", "content": "hi"},
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    {
                        "id": "c1",
                        "name": "lookup",
                        "arguments": {"id": "1"},
                        "requestor": "assistant",
                    }
                ],
            },
            {"role": "tool", "tool_call_id": "c1", "content": '{"ok": true}'},
            {"role": "assistant", "content": "done"},
        ]
    )
    assert structured[1]["tool_calls"][0]["name"] == "lookup"
    assert structured[2]["role"] == "tool"

    inline = normalize_conversation(
        [
            {"role": "user", "content": "hi"},
            {
                "role": "assistant",
                "content": (
                    "Let me check.\n"
                    "<function=get_user_details>\n"
                    "<parameter=user_id>u1</parameter>\n"
                    "</function>\n"
                    '<|tool|>{"name":"Ada"}<|/tool|>\n'
                    "Found you."
                ),
            },
        ]
    )
    roles = [t["role"] for t in inline]
    assert "tool" in roles
    assert any(t.get("tool_calls") for t in inline)


def test_fma_pipeline_stub_llm(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    corpus = []
    for i, source in enumerate(["human", "sim|paired"]):
        corpus.append(
            {
                "primary_key": f"{source}:ep{i}",
                "episode_id": f"ep{i}",
                "source": "human" if source == "human" else "sim",
                "source_id": source if source == "human" else "sim|paired",
                "arm": "paired" if "sim" in source else None,
                "variant": None,
                "domain": "airline",
                "task_id": "0",
                "task_description": "book flight",
                "success_criteria": "book a flight",
                "reward": 0.0,
                "is_success": False,
                "reward_report": {
                    "reward": 0.0,
                    "failed": [{"kind": "db", "passed": False}],
                }
                if "sim" in source
                else None,
                "conversation": [
                    {"role": "user", "content": "book please"},
                    {"role": "assistant", "content": "I refuse for no reason."},
                ],
                "failed": True,
            }
        )
    # pad so sample n=2 works and remainder tagging has extras
    for i in range(2, 6):
        corpus.append(
            {
                **corpus[1],
                "primary_key": f"sim|paired:ep{i}",
                "episode_id": f"ep{i}",
                "source_id": "sim|paired",
                "source": "sim",
            }
        )
    write_jsonl(run_dir / "corpus.jsonl", corpus)

    def stub_chat(messages, **kwargs):
        return json.dumps(
            {
                "label": "Unauthorized Refusal",
                "explanation": "Agent refused without checking policy.",
                "turn_indices": [1],
                "excerpt": "I refuse for no reason.",
            }
        )

    sampled = sample_batch(run_dir, batch=1, n=2, seed=0)
    assert sampled["n_sampled"] == 2
    proposed = propose_batch(run_dir, batch=1, chat_fn=stub_chat)
    assert proposed["n_proposals"] == 2
    props = list((run_dir / "batches/01/proposals.jsonl").read_text().splitlines())
    assert json.loads(props[0])["turn_indices"] == [1]

    decisions = []
    for line in props:
        row = json.loads(line)
        decisions.append(
            {
                "primary_key": row["primary_key"],
                "label": "Unauthorized Refusal",
                "description": "Agent refuses a permitted request.",
                "notes": "ok",
            }
        )
    finished = finish_batch(run_dir, 1, decisions)
    assert finished["n_decisions"] == 2
    taxonomy = load_taxonomy(run_dir)
    assert any(
        m["name"] == "Unauthorized Refusal" and m.get("examples") for m in taxonomy
    )

    tagged = tag_remainder(run_dir, chat_fn=stub_chat)
    assert tagged["n_from_review"] == 2
    assert tagged["n_classified"] == len(corpus) - 2

    report = compute_distributions(run_dir, human_source="human")
    assert "sim|paired" in report["comparisons"]
    cmp = report["comparisons"]["sim|paired"]
    assert 0.0 <= float(cmp["sim_error_rate"]) <= 1.0
    assert cmp["tvd_attribution"] is not None
    assert 0.0 <= float(cmp["tvd_attribution"]) <= 1.0
    assert cmp["tvd_agent"] is not None
    assert 0.0 <= float(cmp["tvd_agent"]) <= 1.0


def test_tag_resumes_from_tagged_jsonl(tmp_path: Path) -> None:
    """Second ``cue-fma tag`` after reinjest only classifies newly added failures."""

    run_dir = tmp_path / "fma"
    run_dir.mkdir()
    save_taxonomy(
        run_dir, [{"name": "Unauthorized Refusal", "description": "x", "examples": []}]
    )
    write_jsonl(
        run_dir / "corpus.jsonl",
        [
            {
                "primary_key": "sim:a",
                "episode_id": "a",
                "source_id": "sim",
                "task_description": "t",
                "success_criteria": "c",
                "conversation": [{"role": "user", "content": "hi"}],
            },
            {
                "primary_key": "sim:b",
                "episode_id": "b",
                "source_id": "sim",
                "task_description": "t",
                "success_criteria": "c",
                "conversation": [{"role": "user", "content": "hi"}],
            },
        ],
    )
    calls = {"n": 0}

    def stub_chat(messages, **kwargs):
        calls["n"] += 1
        return json.dumps(
            {
                "label": "Unauthorized Refusal",
                "explanation": "x",
                "turn_indices": [0],
                "excerpt": "hi",
            }
        )

    first = tag_remainder(run_dir, chat_fn=stub_chat)
    assert first["n_classified"] == 2
    assert first["n_reused"] == 0
    assert calls["n"] == 2

    # Reinjest: keep a/b, add c.
    write_jsonl(
        run_dir / "corpus.jsonl",
        [
            {
                "primary_key": "sim:a",
                "episode_id": "a",
                "source_id": "sim",
                "task_description": "t",
                "success_criteria": "c",
                "conversation": [{"role": "user", "content": "hi"}],
            },
            {
                "primary_key": "sim:b",
                "episode_id": "b",
                "source_id": "sim",
                "task_description": "t",
                "success_criteria": "c",
                "conversation": [{"role": "user", "content": "hi"}],
            },
            {
                "primary_key": "sim:c",
                "episode_id": "c",
                "source_id": "sim",
                "task_description": "t",
                "success_criteria": "c",
                "conversation": [{"role": "user", "content": "hi"}],
            },
        ],
    )
    second = tag_remainder(run_dir, chat_fn=stub_chat)
    assert second["n_classified"] == 1
    assert second["n_reused"] == 2
    assert second["n_tagged"] == 3
    assert calls["n"] == 3  # only one new LLM call

    forced = tag_remainder(run_dir, chat_fn=stub_chat, force=True)
    assert forced["n_classified"] == 3
    assert forced["n_reused"] == 0
    assert calls["n"] == 6


def test_tag_relabel_only_matching_labels(tmp_path: Path) -> None:
    run_dir = tmp_path / "fma"
    run_dir.mkdir()
    save_taxonomy(
        run_dir,
        [
            {"name": "Premature User Stop", "description": "stop", "examples": []},
            {"name": "Wrong Action Parameter", "description": "param", "examples": []},
        ],
    )
    write_jsonl(
        run_dir / "corpus.jsonl",
        [
            {
                "primary_key": "sim:a",
                "episode_id": "a",
                "source_id": "sim",
                "task_description": "t",
                "success_criteria": "c",
                "conversation": [{"role": "user", "content": "hi"}],
            },
            {
                "primary_key": "sim:b",
                "episode_id": "b",
                "source_id": "sim",
                "task_description": "t",
                "success_criteria": "c",
                "conversation": [{"role": "user", "content": "hi"}],
            },
        ],
    )
    write_jsonl(
        run_dir / "tagged.jsonl",
        [
            {
                "primary_key": "sim:a",
                "label": "Critical User Simulator Error",
                "origin": "tag",
            },
            {
                "primary_key": "sim:b",
                "label": "Wrong Action Parameter",
                "origin": "tag",
            },
        ],
    )
    calls = {"n": 0}

    def stub_chat(messages, **kwargs):
        calls["n"] += 1
        return json.dumps(
            {
                "label": "Premature User Stop",
                "explanation": "early stop",
                "turn_indices": [0],
                "excerpt": "yes ###STOP###",
            }
        )

    out = tag_remainder(
        run_dir,
        chat_fn=stub_chat,
        relabel=["Critical User Simulator Error"],
    )
    assert out["n_classified"] == 1
    assert out["n_reused"] == 1
    assert calls["n"] == 1
    by_key = {
        r["primary_key"]: r["label"] for r in read_jsonl(str(run_dir / "tagged.jsonl"))
    }
    assert by_key == {
        "sim:a": "Premature User Stop",
        "sim:b": "Wrong Action Parameter",
    }


def test_ingest_sim_prefers_full_conversation(tmp_path: Path) -> None:
    path = tmp_path / "rollout.tau2.jsonl"
    write_jsonl(
        path,
        [
            {
                "episode_id": "airline_1",
                "task_id": "1",
                "dataset": "airline",
                "task_description": "x",
                "rollout_conversation": [
                    {"role": "user", "content": "hi"},
                    {"role": "assistant", "content": "stripped only"},
                ],
                "metadata": {
                    "arm": "paired",
                    "variant": "base_api",
                    "is_success": False,
                    "reward": 0.0,
                    "tau2_domain": "airline",
                    "tau2_task_id": "1",
                    "reward_report": {
                        "reward": 0.0,
                        "failed": [{"kind": "db", "passed": False}],
                    },
                    "full_conversation": [
                        {"role": "user", "content": "hi"},
                        {
                            "role": "assistant",
                            "content": "",
                            "tool_calls": [
                                {
                                    "id": "c1",
                                    "name": "get_user_details",
                                    "arguments": {"user_id": "u"},
                                    "requestor": "assistant",
                                }
                            ],
                        },
                        {"role": "tool", "tool_call_id": "c1", "content": "{}"},
                        {"role": "assistant", "content": "done"},
                    ],
                },
            }
        ],
    )
    rows = ingest_sim(path, criteria={})
    assert len(rows) == 1
    assert rows[0]["reward_report"]["failed"][0]["kind"] == "db"
    assert any(t.get("tool_calls") for t in rows[0]["conversation"])


def test_ingest_human_matches_task_by_content_not_id(tmp_path: Path) -> None:
    """tau-usi ``airline_0`` is tau2 airline task ``20``, so ids must not drive the join."""

    domains = tmp_path / "data" / "tau2" / "domains" / "airline"
    domains.mkdir(parents=True)
    (domains / "tasks.json").write_text(
        json.dumps(
            [
                {
                    "id": "0",
                    "description": {"purpose": "Refuse a disallowed cancellation."},
                    "evaluation_criteria": {
                        "nl_assertions": ["Agent refuses the cancellation."]
                    },
                    "user_scenario": {
                        "instructions": {
                            "known_info": "You are Emma Kim. Your user id is emma_kim_9957.",
                            "reason_for_call": "You want to cancel your flight reservation.",
                            "task_instructions": "You insist you were told insurance was unnecessary.",
                        }
                    },
                },
                {
                    "id": "20",
                    "description": {
                        "purpose": "Book a flight with time and payment constraints."
                    },
                    "evaluation_criteria": {
                        "nl_assertions": ["Agent books JFK to SEA on 2024-05-20."]
                    },
                    "user_scenario": {
                        "instructions": {
                            "known_info": "You are Mia Li. Your user id is mia_li_3668.",
                            "reason_for_call": "You want to fly from New York to Seattle on May 20 (one way).",
                            "task_instructions": "You do not want to fly before 11am est. You want economy.",
                        }
                    },
                },
            ]
        ),
        encoding="utf-8",
    )

    path = tmp_path / "normalized.jsonl"
    write_jsonl(
        path,
        [
            {
                "episode_id": "airline_0",
                "task_id": "airline_0",
                "dataset": "tau_usi",
                "task_description": (
                    "You are role-playing as a user interacting with an agent.\n\n"
                    "Your user id is mia_li_3668. You want to fly from New York to Seattle on "
                    "May 20 (one way). You do not want to fly before 11am est. You want economy.\n\n"
                    "**You may start with: hi"
                ),
                "real_conversation": [{"role": "user", "content": "hi"}],
                "metadata": {"domain": "customer_service"},
            }
        ],
    )

    criteria = load_task_criteria(str(domains.parent))
    rows = ingest_human(
        path,
        rewards={"airline_0": 0.0},
        criteria=criteria,
        tasks_dir=str(domains.parent),
    )
    assert len(rows) == 1
    assert rows[0]["domain"] == "airline"
    assert rows[0]["task_id"] == "20"
    assert "JFK to SEA" in rows[0]["success_criteria"]


def test_ingest_sim_source_id_separates_simulators(tmp_path: Path) -> None:
    """Same method, different simulator LLM: distinct source_id and primary_key."""

    def _write(experiment: str, simulator: str) -> Path:
        path = (
            tmp_path
            / "outputs/rollouts"
            / experiment
            / "tau2/customer-service/baseline/ppol"
            / simulator
            / "rollout.tau2.jsonl"
        )
        path.parent.mkdir(parents=True, exist_ok=True)
        write_jsonl(
            path,
            [
                {
                    "episode_id": "airline_1",
                    "task_id": "1",
                    "dataset": "airline",
                    "rollout_conversation": [
                        {"role": "user", "content": "hi"},
                        {"role": "assistant", "content": "no"},
                    ],
                    "metadata": {
                        "arm": "as_is",
                        "variant": "baseline:ppol",  # identical across simulators
                        "is_success": False,
                        "reward": 0.0,
                        "tau2_domain": "airline",
                        "tau2_task_id": "1",
                    },
                }
            ],
        )
        return path

    llama = ingest_sim(_write("ppol", "llama"), criteria={})
    gpt = ingest_sim(_write("ppol", "gpt"), criteria={})
    assert llama[0]["source_id"] == "ppol-llama|as_is"
    assert gpt[0]["source_id"] == "ppol-gpt|as_is"
    assert llama[0]["primary_key"] != gpt[0]["primary_key"]


def test_ingest_sim_source_id_skips_seed_wrapper(tmp_path: Path) -> None:
    """Multiseed layout must not collapse methods onto ``seed-N-<sim>``."""

    path = (
        tmp_path
        / "outputs/rollouts/seed-0/baseline-seed-0"
        / "tau2/customer-service/baseline/realusersim/gemini/rollout.tau2.jsonl"
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    write_jsonl(
        path,
        [
            {
                "episode_id": "airline_1",
                "task_id": "1",
                "dataset": "airline",
                "rollout_conversation": [
                    {"role": "user", "content": "hi"},
                    {"role": "assistant", "content": "no"},
                ],
                "metadata": {
                    "arm": "as_is",
                    "variant": "baseline:realusersim",
                    "is_success": False,
                    "reward": 0.0,
                    "tau2_domain": "airline",
                    "tau2_task_id": "1",
                },
            }
        ],
    )
    rows = ingest_sim(path, criteria={})
    assert rows[0]["source_id"] == "realusersim-gemini|as_is"
    assert rows[0]["seed"] == 0
    assert rows[0]["primary_key"] == "realusersim-gemini|as_is:seed-0:airline_1"


def test_ingest_pools_source_id_across_seeds(tmp_path: Path) -> None:
    """Same method/sim across seeds share source_id; primary_keys stay distinct."""

    def _write(seed: int) -> Path:
        path = (
            tmp_path
            / f"outputs/rollouts/seed-{seed}/ppol"
            / "tau2/customer-service/baseline/ppol/llama/rollout.tau2.jsonl"
        )
        path.parent.mkdir(parents=True, exist_ok=True)
        write_jsonl(
            path,
            [
                {
                    "episode_id": "airline_1",
                    "task_id": "1",
                    "dataset": "airline",
                    "rollout_conversation": [
                        {"role": "user", "content": "hi"},
                        {"role": "assistant", "content": "no"},
                    ],
                    "metadata": {
                        "arm": "as_is",
                        "variant": "baseline:ppol",
                        "is_success": False,
                        "reward": 0.0,
                        "tau2_domain": "airline",
                        "tau2_task_id": "1",
                    },
                }
            ],
        )
        return path

    a = ingest_sim(_write(0), criteria={})
    b = ingest_sim(_write(1), criteria={})
    assert a[0]["source_id"] == b[0]["source_id"] == "ppol-llama|as_is"
    assert a[0]["primary_key"] != b[0]["primary_key"]
    assert {a[0]["seed"], b[0]["seed"]} == {0, 1}


def test_ingest_sim_source_id_cue_general(tmp_path: Path) -> None:
    path = (
        tmp_path
        / "outputs/rollouts/seed-1/cue-general-seed-1"
        / "tau2/customer-service/cue/general/gpt/rollout.tau2.jsonl"
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    write_jsonl(
        path,
        [
            {
                "episode_id": "airline_1",
                "task_id": "1",
                "dataset": "airline",
                "rollout_conversation": [
                    {"role": "user", "content": "hi"},
                    {"role": "assistant", "content": "no"},
                ],
                "metadata": {
                    "arm": "as_is",
                    "variant": "cue:general",
                    "is_success": False,
                    "reward": 0.0,
                    "tau2_domain": "airline",
                    "tau2_task_id": "1",
                },
            }
        ],
    )
    rows = ingest_sim(path, criteria={})
    assert rows[0]["source_id"] == "cue-general-gpt|as_is"


def test_total_variation_half_l1() -> None:
    assert total_variation([1.0, 0.0], [0.0, 1.0]) == 1.0
    assert total_variation([0.5, 0.5], [0.5, 0.5]) == 0.0


def test_tvd_attribution_and_agent_from_tagged(tmp_path: Path) -> None:
    """CLI metrics: sim error %, 3-way attribution TVD, agent-mode TVD."""

    run_dir = tmp_path / "fma"
    run_dir.mkdir()
    save_taxonomy(
        run_dir,
        [
            {"name": "Other Simulator Error", "description": "user", "examples": []},
            {"name": "Environment Error", "description": "env", "examples": []},
            {"name": "Wrong Action Parameter", "description": "agent", "examples": []},
            {"name": "Policy-Forbidden Action", "description": "agent", "examples": []},
        ],
    )
    write_jsonl(
        run_dir / "tagged.jsonl",
        [
            {"source_id": "human", "label": "Other Simulator Error"},
            {"source_id": "human", "label": "Environment Error"},
            {"source_id": "human", "label": "Wrong Action Parameter"},
            {"source_id": "human", "label": "Policy-Forbidden Action"},
            {"source_id": "sim|paired", "label": "Other Simulator Error"},
            {"source_id": "sim|paired", "label": "Other Simulator Error"},
            {"source_id": "sim|paired", "label": "Wrong Action Parameter"},
            {"source_id": "sim|paired", "label": "Policy-Forbidden Action"},
        ],
    )
    report = compute_distributions(run_dir, human_source="human")
    human = report["human"]
    cmp = report["comparisons"]["sim|paired"]

    assert human["sim_error_rate"] == 0.25
    assert cmp["sim_error_rate"] == 0.5
    # Human attribution [0.25, 0.25, 0.50]; sim [0.50, 0.00, 0.50]
    assert human["attribution_proportions"] == {
        "user_error": 0.25,
        "env_error": 0.25,
        "agent_error": 0.5,
    }
    assert cmp["attribution_proportions"] == {
        "user_error": 0.5,
        "env_error": 0.0,
        "agent_error": 0.5,
    }
    assert cmp["tvd_attribution"] == pytest.approx(0.25)
    # Agent-only mix: human equal Wrong/Policy; sim equal Wrong/Policy → TVD 0.
    assert cmp["tvd_agent"] == pytest.approx(0.0)


def test_rename_and_merge_failure_modes_rewrites_labels(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    batch = run_dir / "batches" / "01"
    batch.mkdir(parents=True)
    save_taxonomy(
        run_dir,
        [
            {
                "name": "Wrong Refund",
                "description": "a",
                "examples": [{"snippet": "x"}],
            },
            {"name": "Bad Charge", "description": "b", "examples": [{"snippet": "y"}]},
            {"name": "Other", "description": "c", "examples": []},
        ],
    )
    write_jsonl(
        batch / "proposals.jsonl",
        [
            {"primary_key": "a", "label": "Wrong Refund"},
            {"primary_key": "b", "label": "Bad Charge"},
        ],
    )
    write_jsonl(
        batch / "decisions.jsonl",
        [
            {
                "primary_key": "a",
                "label": "Wrong Refund",
                "proposed_label": "Wrong Refund",
            },
        ],
    )
    write_jsonl(
        run_dir / "tagged.jsonl",
        [
            {"primary_key": "t1", "label": "Wrong Refund"},
            {"primary_key": "t2", "label": "Bad Charge"},
        ],
    )

    renamed = rename_failure_mode(run_dir, old="Wrong Refund", new="Incorrect Refund")
    names = {m["name"] for m in renamed["taxonomy"]}
    assert "Incorrect Refund" in names
    assert "Wrong Refund" not in names
    assert renamed["remapped"]["proposals"] == 1
    assert renamed["remapped"]["decisions"] == 1
    assert renamed["remapped"]["tagged"] == 1

    merged = merge_failure_modes(
        run_dir,
        sources=["Incorrect Refund", "Bad Charge"],
        target="Payment Error",
        description="Payment mishandling",
    )
    names = [m["name"] for m in merged["taxonomy"]]
    assert names.count("Payment Error") == 1
    assert "Incorrect Refund" not in names
    assert "Bad Charge" not in names
    props = read_jsonl(str(batch / "proposals.jsonl"))
    assert {r["label"] for r in props} == {"Payment Error"}
    tagged = read_jsonl(str(run_dir / "tagged.jsonl"))
    assert {r["label"] for r in tagged} == {"Payment Error"}
    mode = next(m for m in merged["taxonomy"] if m["name"] == "Payment Error")
    assert mode["description"] == "Payment mishandling"
    assert len(mode["examples"]) == 2


def test_prune_empty_modes(tmp_path: Path) -> None:
    from cue_training.evaluation.fma.review import prune_empty_modes

    run_dir = tmp_path / "run"
    batch = run_dir / "batches" / "01"
    batch.mkdir(parents=True)
    save_taxonomy(
        run_dir,
        [
            {"name": "Used", "description": "", "examples": []},
            {"name": "Empty", "description": "gone", "examples": []},
            {"name": "Uncategorized", "description": "", "examples": []},
        ],
    )
    write_jsonl(batch / "proposals.jsonl", [{"primary_key": "a", "label": "Used"}])
    result = prune_empty_modes(run_dir)
    names = {m["name"] for m in result["taxonomy"]}
    assert names == {"Used", "Uncategorized"}
    assert result["removed"] == ["Empty"]
    assert result["counts"]["Used"] == 1


def test_review_uncategorized_updates_tagged(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    (run_dir / "batches").mkdir()
    save_taxonomy(
        run_dir,
        [
            {"name": "Wrong Action", "description": "bad tool", "examples": []},
            {"name": "Uncategorized", "description": "no fit", "examples": []},
        ],
    )
    corpus = [
        {
            "primary_key": "human:a",
            "episode_id": "a",
            "source_id": "human",
            "source": "human",
            "conversation": [{"role": "user", "content": "hi", "turn_index": 0}],
            "task_description": "t",
            "success_criteria": "c",
        },
        {
            "primary_key": "sim:b",
            "episode_id": "b",
            "source_id": "sim|paired",
            "source": "sim",
            "conversation": [{"role": "user", "content": "yo", "turn_index": 0}],
            "task_description": "t",
            "success_criteria": "c",
        },
    ]
    write_jsonl(run_dir / "corpus.jsonl", corpus)
    write_jsonl(
        run_dir / "tagged.jsonl",
        [
            {
                "primary_key": "human:a",
                "episode_id": "a",
                "source_id": "human",
                "label": "Uncategorized",
                "explanation": "no mode fit",
                "turn_indices": [0],
                "excerpt": "hi",
                "origin": "tag",
            },
            {
                "primary_key": "sim:b",
                "episode_id": "b",
                "source_id": "sim|paired",
                "label": "Wrong Action",
                "explanation": "ok",
                "turn_indices": [0],
                "excerpt": "yo",
                "origin": "tag",
            },
        ],
    )
    prepared = sample_uncategorized_batch(run_dir, batch=1)
    assert prepared["n_sampled"] == 1
    assert prepared["n_pool"] == 1
    props = list((run_dir / "batches/01/proposals.jsonl").read_text().splitlines())
    assert json.loads(props[0])["label"] == "Uncategorized"

    finished = finish_batch(
        run_dir,
        1,
        [
            {
                "primary_key": "human:a",
                "label": "Wrong Action",
                "explanation": "actually wrong action",
                "turn_indices": [0],
                "excerpt": "hi",
            }
        ],
    )
    assert finished["n_decisions"] == 1
    assert finished["n_tagged_updated"] == 1
    from cue_training.evaluation.common.io import read_jsonl

    tagged_rows = {
        r["primary_key"]: r for r in read_jsonl(str(run_dir / "tagged.jsonl"))
    }
    assert tagged_rows["human:a"]["label"] == "Wrong Action"
    assert tagged_rows["human:a"]["origin"] == "review"
    assert tagged_rows["sim:b"]["label"] == "Wrong Action"

    # Already decided Uncategorized keys are skipped on the next pass.
    again = sample_uncategorized_batch(run_dir, batch=2)
    assert again["n_sampled"] == 0
