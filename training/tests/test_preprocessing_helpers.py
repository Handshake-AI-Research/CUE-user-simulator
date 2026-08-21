"""Preprocessing helpers: style prompts, dispatch, delexicalization, truncation, augmentation,
streaming, and eval-profile conversion."""

from __future__ import annotations

import json
import random
from pathlib import Path

from cue_training.augment import dropout_user_turns
from cue_training.data.streaming import (
    RoundRobinStreamingDataset,
    discover_jsonl_files,
    write_jsonl_atomic,
    write_streaming_index,
)
from cue_training.preprocessing.delexicalize import delexicalize_manual, delexicalize_text
from cue_training.preprocessing.dispatch import LeastInflightPool, build_default_router
from cue_training.preprocessing.eval_profiles import (
    REALUSERSIM_VARIANT,
    _prism_cases,
    _simarena_cases,
    case_to_record,
    manual_to_persona,
    select_cases,
    write_persona_sidecar,
)
from cue_training.preprocessing.llm import _extract_json_object, estimate_tokens, truncate_messages
from cue_training.preprocessing.style_augment import build_style_messages


def test_style_prompt_requests_surface_habits_only():
    messages = build_style_messages(
        turns=[
            {"role": "user", "content": "ok fix it", "turn_id": "t0"},
            {"role": "assistant", "content": "done", "turn_id": "t1"},
        ],
        n_style=3,
    )
    system = messages[0]["content"]
    assert "SURFACE WRITING STYLE" in system
    assert "lowercase" in system.lower()
    assert "dialogue-policy" in system.lower()
    assert "Extract 3 style commands" in messages[1]["content"]


def test_least_inflight_picks_idle_replica() -> None:
    pool = LeastInflightPool(["a", "b"])
    with pool.acquire() as first:
        assert first == "a"
        with pool.acquire() as second:
            assert second == "b"
    with pool.acquire() as again:
        assert again == "a"


def test_router_matches_llama_and_qwen() -> None:
    router = build_default_router()
    with router.api_base_for("hosted_vllm/meta-llama/Llama-3.1-8B-Instruct") as base:
        assert base == "http://127.0.0.1:8000/v1"
    with router.api_base_for("hosted_vllm/Qwen/Qwen3-8B") as base:
        assert base == "http://127.0.0.1:8001/v1"
    with router.api_base_for("gpt-5.4-mini", fallback=None) as base:
        assert base is None


def test_delexicalize_replaces_urls_and_numbers():
    text = delexicalize_text('visit https://example.com and pay 42 dollars for "Acme Corp"')
    assert "<URL>" in text
    assert "<NUM>" in text
    assert "<ENTITY>" in text


def test_delexicalize_manual():
    manual = {
        "schema_version": 1,
        "commands": [
            {
                "text": "Ask about ticket 123",
                "kind": "human_contrast",
                "form": "general",
                "examples": ["see https://x.test"],
            }
        ],
    }
    out = delexicalize_manual(manual)
    assert "<NUM>" in out["commands"][0]["text"]
    assert "<URL>" in out["commands"][0]["examples"][0]


def test_truncate_messages_keeps_recent_within_budget() -> None:
    messages = [
        {"role": "system", "content": "sys"},
        {"role": "user", "content": "A" * 5000},
        {"role": "assistant", "content": "B" * 5000},
        {"role": "user", "content": "please continue"},
    ]
    out = truncate_messages(messages, max_input_tokens=2000, max_tokens=128)
    assert out[0]["role"] == "system"
    assert out[-1]["content"] == "please continue"
    total = sum(estimate_tokens(m["content"]) for m in out)
    assert total < 2000


def test_truncate_messages_shrinks_oversized_system() -> None:
    messages = [
        {"role": "system", "content": "S" * 20_000},
        {"role": "user", "content": "please continue with the draft"},
    ]
    out = truncate_messages(messages, max_input_tokens=4096, max_tokens=512)
    assert out[0]["role"] == "system"
    assert out[-1]["role"] == "user"
    total = sum(estimate_tokens(m["content"]) for m in out)
    assert total <= 4096 - 512
    assert len(out[0]["content"]) < 20_000


def test_extract_json_strips_fences_and_trailing_commas() -> None:
    text = """```json
{"schema_version": 1, "commands": [{"text": "Be terse", "kind": "sim_contrast", "form": "general", "examples": ["ok"]},],}
```"""
    data = _extract_json_object(text)
    assert data["schema_version"] == 1
    assert data["commands"][0]["text"] == "Be terse"
    # round-trip sanity
    json.dumps(data)


def _turns():
    return [
        {"role": "system", "content": "task"},
        {"role": "user", "content": "u1"},
        {"role": "assistant", "content": "a1"},
        {"role": "user", "content": "u2"},
        {"role": "assistant", "content": "a2"},
        {"role": "user", "content": "u3"},
        {"role": "assistant", "content": "a3"},
    ]


def test_dropout_user_turns_drops_context_bundles():
    dropped = dropout_user_turns(_turns(), 0.5, rng=random.Random(1))
    assert [turn["content"] for turn in dropped] == ["task", "u1", "a1", "u3", "a3"]


def test_dropout_user_turns_always_retains_first_and_last_user_turns():
    dropped = dropout_user_turns(_turns(), 1.0, rng=random.Random(0))
    assert [turn["content"] for turn in dropped] == ["task", "u1", "a1", "u3", "a3"]


def _manual():
    return {
        "schema_version": 1,
        "commands": [
            {"text": f"Rule {i}", "kind": "sim_contrast" if i < 5 else "human_contrast", "form": "general", "examples": []}
            for i in range(10)
        ],
    }


def test_dataset_wise_round_robin(tmp_path: Path):
    a = tmp_path / "ds_a" / "train.jsonl"
    b = tmp_path / "ds_b" / "train.jsonl"
    write_jsonl_atomic(
        a,
        [
            {
                "id": "a1",
                "dataset": "ds_a",
                "turns": [{"role": "user", "content": "a1"}],
                "persona_manual": _manual(),
            },
            {
                "id": "a2",
                "dataset": "ds_a",
                "turns": [{"role": "user", "content": "a2"}],
                "persona_manual": _manual(),
            },
        ],
    )
    write_jsonl_atomic(
        b,
        [
            {
                "id": "b1",
                "dataset": "ds_b",
                "turns": [{"role": "user", "content": "b1"}],
                "persona_manual": _manual(),
            }
        ],
    )
    files = discover_jsonl_files(tmp_path, split="train")
    assert len(files) == 2
    write_streaming_index(tmp_path, files)
    assert (tmp_path / "streaming_index.json").exists()

    from cue_training.data.streaming import count_from_streaming_index

    assert count_from_streaming_index(tmp_path, files) == 3
    assert count_from_streaming_index(tmp_path, files[:1]) == 2
    assert count_from_streaming_index(tmp_path / "missing", files) is None

    ds = RoundRobinStreamingDataset(tmp_path, split="train", per_source_cap=10, shuffle_buffer=0)
    ids = [r["id"] for r in ds]
    # Round-robin should interleave datasets rather than dump all of a first.
    assert ids[0][0] != ids[1][0] or len(ids) == 1
    assert set(ids) == {"a1", "a2", "b1"}


def test_simarena_annotations_convert_to_eval_cases(tmp_path):
    path = tmp_path / "annotations.json"
    path.write_text(
        json.dumps(
            [
                {
                    "model": "m",
                    "workerId": "w",
                    "document_type": "email",
                    "intent": "request",
                    "user_queries": ["help", "thanks"],
                    "ai_responses": ["sure"],
                }
            ]
        )
    )
    cases = _simarena_cases(path, "writing")
    assert cases[0]["episode_id"] == "m_w_email_request"
    assert cases[0]["real_conversation"] == [
        {"role": "user", "content": "help"},
        {"role": "assistant", "content": "sure"},
        {"role": "user", "content": "thanks"},
    ]


def test_prism_normalized_converts_to_eval_cases(tmp_path):
    path = tmp_path / "normalized.jsonl"
    path.write_text(
        json.dumps(
            {
                "dataset": "prism",
                "episode_id": "c0",
                "task_id": "c0",
                "real_conversation": [
                    {"role": "user", "content": "Hello"},
                    {"role": "assistant", "content": "Hi"},
                    {"role": "user", "content": "What is 2+2?"},
                ],
                "metadata": {"domain": "diversity"},
            }
        )
        + "\n"
    )
    cases = _prism_cases(path)
    assert len(cases) == 1
    assert cases[0]["episode_id"] == "c0"
    rec = case_to_record(cases[0])
    assert rec["dataset"] == "prism"
    assert rec["id"] == "c0"
    assert [t["role"] for t in rec["turns"]] == ["user", "assistant", "user"]
    assert rec["metadata"]["domain"] == "diversity"


def test_select_cases_caps_and_requires_user_turn() -> None:
    cases = [
        {"episode_id": "a", "real_conversation": [{"role": "assistant", "content": "hi"}]},
        {"episode_id": "b", "real_conversation": [{"role": "user", "content": "help"}]},
        {"episode_id": "c", "real_conversation": [{"role": "user", "content": "ok"}]},
        {"episode_id": "d", "real_conversation": [{"role": "user", "content": "more"}]},
    ]
    selected = select_cases(cases, max_cases=2)
    assert [c["episode_id"] for c in selected] == ["b", "c"]


def test_persona_sidecar_keys_match_runner(tmp_path: Path) -> None:
    profiles = [
        {
            "id": "ep1",
            "persona_manual": {
                "schema_version": 1,
                "commands": [
                    {
                        "text": "Be terse.",
                        "kind": "sim_contrast",
                        "form": "general",
                        "examples": ["ok"],
                        "evidence_turn_ids": [],
                    },
                    {
                        "text": "Use lowercase fragments.",
                        "kind": "style",
                        "form": "general",
                        "examples": ["ok"],
                        "evidence_turn_ids": [],
                    },
                ],
            },
        },
        {
            "id": "ep2",
            "persona_manual": {
                "schema_version": 1,
                "commands": [
                    {
                        "text": "Ask clarifying questions.",
                        "kind": "human_contrast",
                        "form": "general",
                        "examples": ["which one?"],
                        "evidence_turn_ids": [],
                    }
                ],
            },
        },
    ]
    path = tmp_path / "personas.jsonl"
    write_persona_sidecar(path, profiles, seed=0)
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    assert {r["variant"] for r in rows} == {REALUSERSIM_VARIANT}
    assert {r["arm"] for r in rows} == {"paired", "sample_shuffled"}
    assert all(r.get("persona_origin_episode_id") for r in rows)
    persona = manual_to_persona(profiles[0]["persona_manual"])
    assert persona["commands"] == ["Be terse.", "Use lowercase fragments."]
    assert persona["examples"] == ["ok"]
    assert "General behavior:" in persona["manual_sections"]
    assert "Writing style:\n- Use lowercase fragments." in persona["manual_sections"]
