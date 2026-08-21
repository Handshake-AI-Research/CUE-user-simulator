import json

import numpy as np
import pytest

from cue_training.evaluation.common.commands import render_command_block, validate_command_block
from cue_training.evaluation.common.embeddings import load_embeddings
from cue_training.evaluation.common.io import read_jsonl, write_jsonl
from cue_training.evaluation.common.shuffle import build_shuffle_map, domain_of
from cue_training.evaluation.common.mirror_prompts import (
    trim_control_message,
    user_proxy_system_prompt,
    user_proxy_task_block,
)
from cue_training.evaluation.common.rollout import clean_user_message, rollout_record_from_conversation


def test_jsonl_and_embedding_loading(tmp_path):
    jsonl_path = tmp_path / "nested" / "rows.jsonl"
    rows = [{"value": 1}, {"value": "é"}]
    write_jsonl(str(jsonl_path), rows)
    assert read_jsonl(str(jsonl_path)) == rows

    embeddings_path = tmp_path / "embeddings.json"
    embeddings_path.write_text(
        json.dumps(
            [
                {
                    "user_id": "user",
                    "session_id": "session",
                    "trajectory_embedding": [3, 4],
                    "metadata": {"domain": "test", "episode_id": "episode"},
                }
            ]
        )
    )
    by_key, means = load_embeddings(str(embeddings_path))
    # Native encoder scale: the decoder consumes these as its LayerNorm'd bottleneck.
    assert np.allclose(by_key["episode"], [3, 4])
    assert np.allclose(means["test"], [3, 4])


def test_prompt_rollout_and_command_helpers_preserve_behavior():
    task = user_proxy_task_block(task_description="Book it", domain="travel", persona="terse")
    assert task == "Task description: Book it.\nDomain or topic: travel.\nPersona hints: terse."
    assert task in user_proxy_system_prompt(
        task_description="Book it", domain="travel", persona="terse"
    )

    assert clean_user_message("assistant\n\nUser: hello ### done", return_stop=True) == (
        "hello",
        True,
    )
    assert clean_user_message('User: "hello"') == "hello"
    # Decoder path: think wrappers stripped, mid-turn ### still ends the episode.
    assert trim_control_message("Thanks, all set! ###STOP###", return_stop=True) == (
        "Thanks, all set!",
        True,
    )
    assert trim_control_message("<think>plan</think>\nOk ###", return_stop=True) == ("Ok", True)
    assert trim_control_message("still going", return_stop=True) == ("still going", False)
    record = rollout_record_from_conversation(
        {"task_id": "1", "metadata": {"domain": "test"}, "arm": "paired"},
        [{"role": "assistant", "content": "Hi", "raw_data": {"cue_command": "brief"}}],
    )
    assert record["episode_id"] == "1"
    assert record["metadata"]["rollout_kind"] == "paired"
    assert record["rollout_conversation"][0]["metadata"]["cue_command"] == "brief"

    block = validate_command_block({"commands": ["Be brief", "be brief", {"command": "Ask questions"}]})
    assert block == {
        "commands": [
            {"command": "Be brief", "examples": []},
            {"command": "Ask questions", "examples": []},
        ]
    }
    assert "- Be brief" in render_command_block(block)


class _FakeManualDecoder:
    def __init__(self) -> None:
        import torch

        self.lm = torch.nn.Module()
        embedding = torch.nn.Embedding(1, 1)
        self.lm.get_input_embeddings = lambda: embedding
        self.generate_calls = 0

    def tokenizer(self, _text, **_kwargs):
        import torch

        return {"input_ids": torch.zeros(1, 1, dtype=torch.long)}

    def generate(self, *, bottleneck, sentinel_ids, **_kwargs):
        self.generate_calls += 1
        return ['{"commands": ["Keep replies short."], "examples": []}'] * bottleneck.size(0)


class _FakeManualModel:
    class _Encoder:
        bottleneck_dim = 2

    def __init__(self) -> None:
        self.encoder = self._Encoder()
        self.decoder = _FakeManualDecoder()


def test_decode_manuals_batch_reuses_cached_manuals(monkeypatch):
    """cue general shares one decoder across simulators, so a repeat request (next simulator, or a
    resume) must hit the cache instead of paying the decode again -- unless settings changed."""

    from cue_training.evaluation.common import model_hub, user_sims

    model = _FakeManualModel()
    monkeypatch.setattr(model_hub, "get_decoder", lambda **_kwargs: model)
    monkeypatch.setattr(user_sims, "_MANUAL_CACHE", {})
    monkeypatch.setattr(user_sims, "_MANUAL_SIG", {})
    items = [("paired::ep1", [0.0, 1.0]), ("paired::ep2", [1.0, 0.0])]

    first = user_sims.decode_manuals_batch(decoder_dir="ckpt", items=items, device="cpu")
    assert model.decoder.generate_calls == 1
    assert "Keep replies short." in first["paired::ep1"]

    assert user_sims.decode_manuals_batch(decoder_dir="ckpt", items=items, device="cpu") == first
    assert model.decoder.generate_calls == 1

    user_sims.decode_manuals_batch(
        decoder_dir="ckpt", items=items, device="cpu", temperature=0.7
    )
    assert model.decoder.generate_calls == 2
    user_sims.decode_manuals_batch(decoder_dir="other-ckpt", items=items, device="cpu")
    assert model.decoder.generate_calls == 3

    # Same key, different conditioning (e.g. two benchmarks minting the same arm::episode_id)
    # must not reuse the cached manual.
    user_sims.decode_manuals_batch(
        decoder_dir="ckpt", items=[("paired::ep1", [0.5, 0.5])], device="cpu"
    )
    assert model.decoder.generate_calls == 4


def test_command_adapter_accepts_unified_text_entries_and_caps_examples():
    block = validate_command_block(
        {
            "commands": [
                {
                    "text": "Be concise.",
                    "examples": ["one", "two", "three", "four"],
                },
                {"command": "Ask directly.", "examples": ["five"]},
            ]
        }
    )
    assert block is not None
    rendered = render_command_block(block)
    assert "Be concise." in rendered
    assert "Ask directly." in rendered
    assert '"five"' not in rendered


def test_embedding_loader_accepts_unified_jsonl_export(tmp_path):
    path = tmp_path / "embeddings.jsonl"
    path.write_text(
        json.dumps(
            {
                "trajectory_embedding": [3.0, 4.0],
                "session_id": "session",
                "metadata": {"domain": "airline"},
            }
        )
        + "\n"
    )
    by_key, means = load_embeddings(str(path))
    assert by_key["session"] == pytest.approx([3.0, 4.0])
    assert "airline" in means


def test_build_shuffle_map_is_deterministic_and_within_domain():
    ids = ["airline_1", "airline_2", "retail_1", "retail_2", "retail_3"]
    domains = {
        "airline_1": "airline",
        "airline_2": "airline",
        "retail_1": "retail",
        "retail_2": "retail",
        "retail_3": "retail",
    }
    a = build_shuffle_map(ids, domains=domains, seed=0)
    b = build_shuffle_map(list(reversed(ids)), domains=domains, seed=0)
    assert a == b
    assert set(a) == set(ids)
    for src, dst in a.items():
        assert domains[src] == domains[dst]
        if len([i for i in ids if domains[i] == domains[src]]) > 1:
            assert src != dst


def test_domain_of_infers_tau2_prefixes():
    assert domain_of("airline_12_ann3") == "airline"
    assert domain_of("retail_54_ann2") == "retail"
    assert domain_of("x", explicit="writing") == "writing"
