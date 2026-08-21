import numpy as np
import pytest
import torch

from cue_training.data.schema import COMMAND_SLOT_SENTINEL
from cue_training.refinement.buffer import ReplayBuffer
from cue_training.refinement.data import examples_from_record, future_shuffled_session
from cue_training.refinement.grpo import clipped_grpo_loss, group_advantages
from cue_training.refinement.leakage import paired_gap, target_manual_overlap
from cue_training.refinement.probe import (
    PairwiseOSSHiddenStateProbe,
    binary_auc,
    format_probe_input,
)
from cue_training.refinement.reward_panel import (
    RewardPanel,
    length_penalty,
    normalize_score,
    parse_judge_score,
)
from cue_training.refinement.simulator import build_messages, clean_user_message
from cue_training.refinement.train import (
    _cue_contrastive_rewards,
    _cue_turn_session,
    _generate_manuals,
    _sentinel_ids,
)


def _record():
    return {
        "id": "s1",
        "intent": "repair the device",
        "turns": [
            {"role": "user", "content": "It broke."},
            {"role": "assistant", "content": "What happened?"},
            {"role": "user", "content": "It will not start."},
            {"role": "assistant", "content": "Charge it."},
            {"role": "user", "content": "Already did."},
            {"role": "assistant", "content": "Try a reset."},
            {"role": "user", "content": "How?"},
        ],
    }


def test_examples_have_context_and_leakage_controls():
    examples = list(examples_from_record(_record(), seed=9))
    assert [example["human_turn"] for example in examples] == [
        "It will not start.",
        "Already did.",
        "How?",
    ]
    first = examples[0]
    assert first["task_text"] == "repair the device"
    assert first["history"] == _record()["turns"][:2]
    assert first["target_masked_full_session"][2]["content"] == "<TARGET_USER_TURN>"
    shuffled = future_shuffled_session(_record()["turns"], 1, seed=9, session_id="s1")
    assert [turn["role"] for turn in shuffled] == [turn["role"] for turn in _record()["turns"]]
    assert shuffled == future_shuffled_session(_record()["turns"], 1, seed=9, session_id="s1")


def test_simulator_prompt_and_cleanup_are_pure():
    messages = build_messages(
        [{"role": "assistant", "content": "Can I help?"}],
        task_text="fix it",
        manual={"commands": ["Be terse."], "examples": ["nope"]},
    )
    assert "Be terse." in messages[0]["content"]
    assert messages[-2] == {"role": "assistant", "content": "Can I help?"}
    assert messages[-1] == {
        "role": "user",
        "content": "Continue as the user with your next message only.",
    }
    already_user = build_messages(
        [{"role": "user", "content": "hello"}],
        task_text="fix it",
    )
    assert already_user[-1] == {"role": "user", "content": "hello"}
    assert clean_user_message("<think>secret</think>\nUSER: not yet\nASSISTANT: ok") == "not yet"


def test_reward_pure_functions():
    assert normalize_score(1) == 0
    assert normalize_score(7) == 2 / 3
    assert parse_judge_score("Score: 9") == 7
    assert length_penalty("one two", "one two") == 0
    assert RewardPanel([], audit_interval=3).should_audit(6)
    assert not RewardPanel([], audit_interval=3).should_audit(7)


def test_cue_reward_preserves_owner_and_penalizes_other_identity():
    human = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
    candidates = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
    rewards, positive, separation = _cue_contrastive_rewards(
        candidates,
        human,
        torch.tensor([0, 0]),
        margin=0.1,
        separation_weight=1.0,
    )
    assert torch.allclose(positive, torch.tensor([1.0, 0.0]))
    assert torch.allclose(separation, torch.tensor([0.0, 1.1]))
    assert rewards[0] > rewards[1]
    assert _cue_turn_session(_record()["turns"][:4], "No.") == [
        {"role": "assistant", "content": "Charge it."},
        {"role": "user", "content": "No."},
    ]


def test_replay_buffer_bounds_and_restores_rng(tmp_path):
    buffer = ReplayBuffer(2, seed=4)
    for index in range(3):
        buffer.add(str(index), [{"role": "assistant", "content": str(index)}], "h", "s")
    assert [item.session_id for item in buffer] == ["1", "2"]
    path = tmp_path / "buffer.json"
    buffer.save(path)
    restored = ReplayBuffer.load(path)
    assert list(restored) == list(buffer)
    assert restored.sample(2) == buffer.sample(2)


class _SlotTokenizer:
    def __init__(self):
        self.value = None

    def __call__(self, value, **_kwargs):
        self.value = value
        return {"input_ids": torch.tensor([[7]])}


class _SlotModel:
    command_slot_decode = True

    def __init__(self):
        self.decoder = type("Decoder", (), {"tokenizer": _SlotTokenizer()})()
        self.called = False

    def encode(self, sessions):
        return torch.zeros(len(sessions), 2)

    def sample_command_slot_manuals(self, bottleneck, **_kwargs):
        self.called = True
        return (
            ["Be concise."],
            [{"commands": ["Be concise."], "examples": []}],
            bottleneck,
            torch.ones(1, 2, dtype=torch.long),
            torch.ones(1, 2, dtype=torch.long),
            [(0, 0, 2)],
        )


def test_refinement_routes_slot_models_and_uses_slot_sentinel():
    model = _SlotModel()
    assert _sentinel_ids(model, torch.device("cpu")).tolist() == [7]
    assert model.decoder.tokenizer.value == COMMAND_SLOT_SENTINEL
    result = _generate_manuals(
        model,
        [[{"role": "user", "content": "hello"}]],
        sentinel_ids=torch.tensor([7]),
        max_new_tokens=512,
        temperature=0.9,
    )
    assert model.called
    assert result[-1] == [(0, 0, 2)]


def _bare_probe(seed=0):
    probe = PairwiseOSSHiddenStateProbe.__new__(PairwiseOSSHiddenStateProbe)
    probe._rng = np.random.default_rng(seed)
    return probe


def test_pairwise_template_is_exact():
    assert format_probe_input("past", "human", "sim") == (
        "[History]\npast\n\n[Response A]\nhuman\n[Response B]\nsim\n\n"
        "Which is more likely to have been produced by the person in the conversation?\nAnswer:"
    )


def test_training_order_is_seeded_and_balanced_after_truncation():
    examples = [(f"h{i}", f"human{i}", f"sim{i}") for i in range(8)]
    prompts, labels = _bare_probe(7)._training_rows(examples, max_per_class=3)
    repeated_prompts, repeated_labels = _bare_probe(7)._training_rows(examples, max_per_class=3)

    assert prompts == repeated_prompts
    assert labels.tolist() == repeated_labels.tolist()
    assert labels.sum() == 3
    assert len(labels) == 6


def test_candidate_score_derandomizes_probability():
    probe = _bare_probe(4)
    probe.predict_proba_a = lambda history, a, b: 0.8

    result = probe.score_candidate("history", "candidate", "simulator")

    assert result.probability_a_human == 0.8
    expected = 0.8 if result.candidate_is_a else 0.2
    assert result.probability_candidate_human == pytest.approx(expected)


def test_binary_auc_handles_order_and_ties():
    assert binary_auc([0, 0, 1, 1], [0.1, 0.2, 0.8, 0.9]) == 1.0
    assert binary_auc([0, 1], [0.5, 0.5]) == 0.5


class _FakeDecoder:
    def __init__(self, logprob: float):
        self.logprob = torch.tensor(logprob, requires_grad=True)
        self.segments = None

    def manual_token_logprobs(self, *, token_ids, segments=None, **_kwargs):
        self.segments = segments
        return self.logprob.expand_as(token_ids).float()


def test_group_advantages_normalize_each_candidate_group():
    rewards = torch.tensor([0.1, 0.3, 2.0, 2.0])
    advantages = group_advantages(rewards, num_candidates=2).view(2, 2)
    assert torch.allclose(advantages.mean(dim=1), torch.zeros(2), atol=1e-6)
    assert torch.allclose(advantages[1], torch.zeros(2))


def test_clipped_grpo_loss_has_finite_policy_and_kl_terms():
    decoder = _FakeDecoder(-1.0)
    reference = _FakeDecoder(-1.2)
    token_ids = torch.ones(2, 3, dtype=torch.long)
    mask = torch.ones_like(token_ids)
    loss, metrics = clipped_grpo_loss(
        decoder,
        reference,
        bottleneck=torch.zeros(1, 2),
        sentinel_ids=torch.ones(1, dtype=torch.long),
        token_ids=token_ids,
        completion_mask=mask,
        old_token_logprobs=torch.full((2, 3), -1.0),
        rewards=torch.tensor([0.2, 0.8]),
        num_candidates=2,
    )
    assert torch.isfinite(loss)
    assert metrics["kl"] >= 0


def test_clipped_grpo_loss_forwards_slot_segments():
    decoder = _FakeDecoder(-1.0)
    reference = _FakeDecoder(-1.2)
    segments = [(0, 0, 2), (1, 2, 1)]
    loss, _ = clipped_grpo_loss(
        decoder,
        reference,
        bottleneck=torch.zeros(1, 2),
        sentinel_ids=torch.ones(1, dtype=torch.long),
        token_ids=torch.ones(2, 3, dtype=torch.long),
        completion_mask=torch.ones(2, 3, dtype=torch.long),
        old_token_logprobs=torch.full((2, 3), -1.0),
        rewards=torch.tensor([0.2, 0.8]),
        num_candidates=2,
        segments=segments,
    )
    assert torch.isfinite(loss)
    assert decoder.segments == reference.segments == segments


def test_leakage_metrics_detect_positive_gap_and_overlap():
    gap = paired_gap([0.8, 0.7], [0.2, 0.3], bootstrap_samples=100, seed=0)
    assert gap["mean"] > 0
    overlap = target_manual_overlap(
        "Say please send the refund today", "please send the refund today"
    )
    assert overlap["ngram_recall"] > 0
