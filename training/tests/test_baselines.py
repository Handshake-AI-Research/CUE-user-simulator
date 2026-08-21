"""Baseline simulators and policy-merge helpers (USP official prompting, RealUserSim, merge)."""

from __future__ import annotations

from pathlib import Path

import pytest

from cue_training.baselines.common.policy_merge import POLICY_VLLM_METHODS, is_hf_model_dir
from cue_training.baselines.realusersim.simulator import RealUserSimSimulator
from cue_training.baselines.usp.simulator import USPOfficialVLLMSimulator


def test_official_usp_prompt_uses_published_profile_system_message() -> None:
    sim = USPOfficialVLLMSimulator(
        [{"goals": "learn", "communication_style": "brief"}],
        base_url="http://localhost:8002/v1",
        model="wangkevin02/USP",
    )
    messages = sim._messages([
        {"role": "user", "content": "Explain recursion."},
        {"role": "assistant", "content": "A function calls itself."},
    ])
    assert messages[0]["role"] == "system"
    assert "Your profile is:" in messages[0]["content"]
    assert "Your goal" not in messages[0]["content"]
    assert messages[1:] == [
        {"role": "assistant", "content": "Explain recursion."},
        {"role": "user", "content": "A function calls itself."},
    ]


def test_official_usp_folds_task_facts_into_profile_not_second_system() -> None:
    """Tau2 identity must land inside the published profile slot (not extra_system_prompt)."""

    sim = USPOfficialVLLMSimulator(
        [{"goals": "learn", "communication_style": "brief"}],
        base_url="http://localhost:8002/v1",
        model="wangkevin02/USP",
    )
    sim._profile_text = "Implicit user profile:\n- goals: cancel an order\n- communication style: brief"
    sim._task_text = (
        "You are ivan_hernandez_6923 living in San Diego, 92133. "
        "You wonder when your air purifier is arriving."
    )
    messages = sim._messages([{"role": "assistant", "content": "Hi! How can I help you today?"}])
    assert len([m for m in messages if m["role"] == "system"]) == 1
    system = messages[0]["content"]
    assert "Your profile is:" in system
    assert "ivan_hernandez_6923" in system
    assert "92133" in system
    assert system.index("Your profile is:") < system.index("ivan_hernandez_6923")
    assert "Your goal / intent" not in system


def test_usp_first_turn_keeps_injected_persona() -> None:
    sim = USPOfficialVLLMSimulator(
        [{"goals": "pool", "communication_style": "x"}],
        base_url="http://localhost:8002/v1",
        model="wangkevin02/USP",
    )
    injected = "Implicit user profile:\n- goals: change seat\n- communication style: direct"
    sim._profile_text = injected
    posted: list[list[dict]] = []

    def _fake_post(messages, max_tokens):  # noqa: ANN001, ARG001
        posted.append(messages)
        return {"choices": [{"message": {"content": "I need a seat change."}}]}

    sim._post = _fake_post  # type: ignore[method-assign]
    sim._completion_budget = lambda messages: 64  # type: ignore[method-assign]
    out = sim.first_turn("You are sofia_hernandez_1234 in 28243.", {})
    assert out == "I need a seat change."
    system = posted[0][0]["content"]
    assert "change seat" in system
    assert "sofia_hernandez_1234" in system
    assert "pool" not in system  # injected persona, not resampled pool profile



def test_usp_baseline_requires_vllm_url(tmp_path) -> None:
    from cue_training.baselines.common.config import RolloutConfig
    from cue_training.baselines.usp.baseline import USPBaseline

    cfg = RolloutConfig(output_dir=tmp_path, artifacts_dir=tmp_path)
    try:
        USPBaseline().load(cfg)
        raise AssertionError("expected RuntimeError without vLLM URL")
    except RuntimeError as exc:
        assert "vLLM" in str(exc)


def test_realusersim_formats_command_examples_and_anti_normalization():
    from cue_training.baselines.realusersim.format_persona import ANTI_NORMALIZATION

    sim = RealUserSimSimulator(
        [
            {
                "demographics": {
                    "age": {"value": "18-24", "source": "extracted", "confidence": 0.9},
                    "location": {"value": "Hong Kong", "source": "extracted", "confidence": None},
                },
                "additional_background": "Experienced with customer service interactions.",
                "command_examples": [
                    {
                        "command": "Use mixed casing with a tendency towards lowercase.",
                        "examples": ["i wanna share a place", "i want to talk about a tourist city"],
                    },
                ],
            }
        ]
    )
    prompt = sim.augmentation("", {})
    assert "Demographics:" in prompt
    assert "Age: 18-24 (source: extracted)" in prompt
    assert "Command: Use mixed casing" in prompt
    assert '"i wanna share a place"' in prompt
    assert ANTI_NORMALIZATION in prompt
    assert 'Do not "clean up" the writing' in prompt


def test_realusersim_prefers_sectioned_manual():
    sim = RealUserSimSimulator(
        [
            {
                "commands": ["flattened fallback"],
                "manual_sections": (
                    "General behavior:\n- Be direct.\n\n"
                    "Writing style:\n- Use lowercase fragments."
                ),
            }
        ]
    )
    prompt = sim.augmentation("", {})
    assert "Writing style:\n- Use lowercase fragments." in prompt
    assert "flattened fallback" not in prompt


def test_usp_decoding_defaults_match_hf_card():
    sim = USPOfficialVLLMSimulator(
        [{"goals": "learn", "communication_style": "brief"}],
        base_url="http://localhost:8002/v1",
        model="wangkevin02/USP",
    )
    assert sim.max_new_tokens == 4096
    assert sim.repetition_penalty == 1.2


def test_usp_completion_budget_fits_server_window():
    sim = USPOfficialVLLMSimulator(
        [{"goals": "learn", "communication_style": "brief"}],
        base_url="http://localhost:8002/v1",
        model="wangkevin02/USP",
    )
    messages = sim._messages([{"role": "assistant", "content": "hi there"}])

    # Window unknown -> fall back to the observed 4096 window and leave prompt room.
    sim._server_max_len = None
    budget = sim._completion_budget(messages)
    assert budget < 4096

    sim._server_max_len = 4096  # must leave room for the prompt
    budget = sim._completion_budget(messages)
    assert budget < 4096
    approx_prompt = sum(len(m["content"]) // 3 + 8 for m in messages)
    assert budget + approx_prompt < 4096


def test_usp_http_error_parses_window_and_prompt():
    from cue_training.baselines.usp.simulator import _USPHTTPError

    tokens = (
        '{"error":{"message":"This model\'s maximum context length is 4096 tokens. '
        "However, you requested 1704 output tokens and your prompt contains at least "
        '2393 input tokens, for a total of at least 4097 tokens.","code":400}}'
    )
    err = _USPHTTPError(400, tokens)
    assert err.reported_max_len() == 4096
    assert err.reported_prompt_tokens() == 2393

    # The char-based body vLLM emits when 0 input tokens are allowed.
    chars = (
        '{"error":{"message":"This model\'s maximum context length is 4096 tokens. '
        "However, you requested 4096 output tokens and your prompt contains 581 "
        'characters (more than 0 characters, which is the upper bound for 0 input '
        'tokens).","code":400}}'
    )
    err = _USPHTTPError(400, chars)
    assert err.reported_max_len() == 4096
    assert err.reported_prompt_tokens() == 194  # ceil(581 / 3)


def test_userlm_guardrail_helpers():
    from cue_training.baselines.userlm.simulator import _passes_guardrails, _starts_with_banned

    assert _starts_with_banned("I need help")
    assert _starts_with_banned("Here is my question")
    assert not _starts_with_banned("Can you help")
    assert not _passes_guardrails("ok", [], "intent")  # < 3 words
    assert not _passes_guardrails(" ".join(["word"] * 26), [], "intent")
    assert _passes_guardrails("need a cheaper flight option please", [], "book a flight")
    assert not _passes_guardrails(
        "need a cheaper flight option please",
        [{"role": "user", "content": "need a cheaper flight option please"}],
        "book a flight",
    )


def test_is_hf_model_dir(tmp_path: Path):
    assert not is_hf_model_dir(tmp_path)
    (tmp_path / "config.json").write_text("{}", encoding="utf-8")
    assert is_hf_model_dir(tmp_path)


def test_no_local_policy_merge_methods():
    assert POLICY_VLLM_METHODS == frozenset()


def test_ensure_merged_raises(tmp_path: Path):
    from cue_training.baselines.common.policy_merge import ensure_merged_policy

    with pytest.raises(ValueError, match="not supported"):
        ensure_merged_policy("usp", tmp_path)
