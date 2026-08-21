import torch
import torch.nn as nn

from cue_training.data.schema import COMMAND_SLOT_SENTINEL, clean_generated_command
from cue_training.decoder.model import PersonaMemory, UnifiedDecoder
from cue_training.model import (
    UnifiedModel,
    command_slot_assignment,
    command_token_jaccard,
    select_command_threshold,
)
from cue_training.training.train import _collate
from cue_training.evaluation.common.user_sims import _resolve_num_candidates, _slot_diverse_decode_kwargs


def test_select_command_threshold_skips_near_dup_keeps_next():
    kept = ["Use a warm, appreciative tone with frequent thanks."]
    cands = [
        "Use a warm appreciative tone with frequent thanks and softeners.",
        "Keep replies short and direct without elaborate politeness.",
    ]
    assert select_command_threshold(cands, kept, near_dup_jaccard=0.5) == cands[1]


def test_select_command_threshold_keeps_first_when_all_clear():
    kept = ["Stay on the same request across turns."]
    cands = [
        "Keep replies short and direct.",
        "Use occasional typos.",
    ]
    assert select_command_threshold(cands, kept, near_dup_jaccard=0.5) == cands[0]


def test_select_command_threshold_skips_near_duplicates():
    kept = ["Keep replies short and direct."]
    assert (
        select_command_threshold(
            ["Keep replies short and direct please."],
            kept,
            near_dup_jaccard=0.5,
        )
        is None
    )


def test_command_token_jaccard_identity():
    assert command_token_jaccard("a b c", "a b c") == 1.0
    assert command_token_jaccard("a b", "c d") == 0.0


def test_diverse_slots_resolve_uses_one_manual_and_per_slot_samples():
    assert _resolve_num_candidates("diverse_slots", 8) == 1
    temp, kwargs = _slot_diverse_decode_kwargs(
        "diverse_slots", num_candidates=8, temperature=0.0, near_dup_jaccard=0.4
    )
    assert temp == 0.7
    assert kwargs == {"num_candidates_per_slot": 8, "near_dup_jaccard": 0.4}


def test_command_slot_assignment_minimizes_unique_cost():
    cost = torch.tensor(
        [
            [1.0, 4.0, 3.0],
            [3.0, 0.5, 2.0],
            [2.0, 3.0, 0.25],
        ]
    )
    assert command_slot_assignment(cost) == [0, 1, 2]


def test_command_slot_assignment_handles_fewer_commands_than_slots():
    cost = torch.tensor([[4.0, 0.2], [0.1, 5.0], [2.0, 2.0]])
    assert command_slot_assignment(cost) == [1, 0]


def test_command_slots_share_one_text_sentinel():
    assert COMMAND_SLOT_SENTINEL == "<COMMAND_SLOT>\n"


class _LayersOnlyLM(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = nn.Module()
        self.model.layers = nn.ModuleList()


def test_learned_slot_is_appended_as_a_memory_token():
    decoder = UnifiedDecoder(
        lm=_LayersOnlyLM(),
        tokenizer=None,
        memory=PersonaMemory(2, 4, num_tokens=2),
        gated_blocks=nn.ModuleList(),
        slot_embeddings=nn.Embedding(2, 3),
        slot_projection=nn.Linear(3, 4),
        general_command_slots=1,
        user_specific_command_slots=1,
        style_command_slots=0,
        insert_every=1,
        config={},
    )
    bottleneck = torch.zeros(1, 2)
    base = decoder.set_persona(bottleneck)
    general = decoder.set_persona(bottleneck, torch.tensor([0]))
    specific = decoder.set_persona(bottleneck, torch.tensor([1]))
    assert base.shape == (1, 2, 4)
    assert general.shape == (1, 3, 4)
    assert not torch.equal(general[:, 0], specific[:, 0])


def test_clean_generated_command_returns_one_rule():
    assert clean_generated_command("- Keep replies short.\n- Ignore this.") == "Keep replies short."
    assert clean_generated_command("<NO_COMMAND>") == ""


def test_collate_preserves_separate_command_sets():
    batch = [
        {
            "id": "s1",
            "turns": [{"role": "user", "content": "hi"}],
            "persona_manual": {
                "commands": [
                    {
                        "text": "Avoid assistant phrasing.",
                        "kind": "sim_contrast",
                    },
                    {
                        "text": "Use lowercase fragments.",
                        "kind": "human_contrast",
                    },
                ]
            },
        }
    ]
    collated = _collate(batch)
    assert collated["general_commands"] == [["Avoid assistant phrasing."]]
    assert collated["user_specific_commands"] == [["Use lowercase fragments."]]
    assert collated["style_commands"] == [[]]


class _FakeTokenizer:
    eos_token = ""

    def __call__(self, values, **_kwargs):
        rows = values if isinstance(values, list) else [values]
        ids = []
        for value in rows:
            if "SLOT_" in value:
                token = int(value.split("SLOT_", 1)[1].split(">", 1)[0])
            else:
                token = {"first": 1, "second": 2}.get(value, 0)
            ids.append([token])
        return {
            "input_ids": torch.tensor(ids),
            "attention_mask": torch.ones(len(ids), 1, dtype=torch.long),
        }


class _FakeDecoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.tokenizer = _FakeTokenizer()
        self.anchor = nn.Parameter(torch.tensor(0.0))
        self.calls = 0

    def forward(
        self,
        *,
        bottleneck,
        target_ids,
        target_mask,
        sentinel_ids,
        slot_ids=None,
        reduction="mean",
    ):
        self.calls += 1
        losses = (target_ids[:, 0].float() - (slot_ids.float() + 1)).abs()
        losses = losses + self.anchor * 0.0 + bottleneck[:, 0] * 0.0
        return losses if reduction == "none" else losses.mean()

    @staticmethod
    def command_slot_id(head, slot):
        if head == "general":
            return slot
        if head == "specific":
            return 2 + slot
        return 4 + slot


class _SamplingDecoder(_FakeDecoder):
    def sample_slot_commands(
        self,
        *,
        bottleneck,
        slot_ids,
        num_candidates,
        **_kwargs,
    ):
        slot_id = int(slot_ids[0])
        rows = bottleneck.size(0) * num_candidates
        text = {0: "Keep replies short.", 1: "Keep replies short.", 2: "Ask one question."}[
            slot_id
        ]
        return (
            [text] * rows,
            torch.full((rows, 2), slot_id + 1, dtype=torch.long),
            torch.ones(rows, 2, dtype=torch.long),
        )


class _NoopThenCommandDecoder(_FakeDecoder):
    """Greedy decodes to ``<NO_COMMAND>``; sampling at temperature > 0 yields a rule."""

    def __init__(self):
        super().__init__()
        self.generate_calls: list[tuple[int, float]] = []

    def generate(self, *, bottleneck, slot_ids=None, temperature=0.0, **_kwargs):
        self.generate_calls.append((bottleneck.size(0), float(temperature)))
        if temperature > 0:
            return [f"- Rule for slot {int(s)}." for s in slot_ids]
        return ["<NO_COMMAND>"] * bottleneck.size(0)


def _noop_model():
    return UnifiedModel(
        nn.Identity(),
        _NoopThenCommandDecoder(),
        dual_decode=True,
        command_slot_decode=True,
        general_command_slots=2,
        user_specific_command_slots=2,
    )


def test_noop_slots_stay_empty_without_retries():
    model = _noop_model()
    blocks = model.generate_command_slot_blocks(
        torch.zeros(2, 2), head="general", num_slots=2
    )
    assert [b["commands"] for b in blocks] == [[], []]


def test_noop_slots_refill_on_retry():
    model = _noop_model()
    blocks = model.generate_command_slot_blocks(
        torch.zeros(2, 2), head="general", num_slots=2, noop_retries=2
    )
    assert [b["commands"] for b in blocks] == [
        ["Rule for slot 0.", "Rule for slot 1."],
        ["Rule for slot 0.", "Rule for slot 1."],
    ]
    # One greedy pass plus one retry per slot; the retry only covers unfilled rows.
    assert model.decoder.generate_calls == [(2, 0.0), (2, 0.7), (2, 0.0), (2, 0.7)]


def test_slot_decode_kwargs_pass_noop_retries_for_greedy():
    _, kwargs = _slot_diverse_decode_kwargs(
        "greedy", num_candidates=1, temperature=0.0, noop_retries=3, noop_temperature=1.0
    )
    assert kwargs == {
        "num_candidates_per_slot": 1,
        "noop_retries": 3,
        "noop_temperature": 1.0,
    }


def test_rollouts_config_noop_retry_reaches_slot_decode():
    """The rollouts.json values must survive CueConfig -> slot-decode kwargs."""

    from pathlib import Path

    from cue_training.evaluation.rollouts.config import load_rollouts_config

    root = Path(__file__).resolve().parents[1]
    cfg = load_rollouts_config(root / "configs" / "rollouts.json")
    _, kwargs = _slot_diverse_decode_kwargs(
        cfg.cue.decode_mode,
        num_candidates=cfg.cue.decode_num_candidates,
        temperature=cfg.cue.decode_temperature,
        noop_retries=cfg.cue.decode_noop_retries,
        noop_temperature=cfg.cue.decode_noop_temperature,
    )
    assert cfg.cue.decode_noop_retries == 0
    assert kwargs["noop_retries"] == 0
    assert kwargs["noop_temperature"] == cfg.cue.decode_noop_temperature


def test_sample_command_slot_manuals_merges_and_masks_duplicates():
    model = UnifiedModel(
        nn.Identity(),
        _SamplingDecoder(),
        dual_decode=True,
        command_slot_decode=True,
        general_command_slots=2,
        user_specific_command_slots=1,
    )
    texts, manuals, _, token_ids, mask, segments = model.sample_command_slot_manuals(
        torch.zeros(1, 2),
        num_candidates=2,
        max_new_tokens=8,
        temperature=0.9,
    )
    assert texts == ["Keep replies short.\nAsk one question."] * 2
    assert [manual["commands"] for manual in manuals] == [
        ["Keep replies short.", "Ask one question."],
        ["Keep replies short.", "Ask one question."],
    ]
    assert token_ids.shape == mask.shape == (2, 6)
    assert not mask[:, 2:4].any()
    assert segments == [(0, 0, 2), (1, 2, 2), (2, 4, 2)]


def test_command_set_loss_matches_each_command_to_unique_best_slot():
    model = UnifiedModel(
        nn.Identity(),
        _FakeDecoder(),
        dual_decode=True,
        command_slot_decode=True,
        general_command_slots=2,
        user_specific_command_slots=2,
    )
    loss, matches = model._command_set_loss(
        torch.zeros(1, 2),
        [["first", "second"]],
        head="general",
        num_slots=2,
    )
    assert float(loss.detach()) == 0.0
    assert [(slot, command) for _, slot, command, _ in matches] == [
        (0, "first"),
        (1, "second"),
    ]
    assert model.decoder.calls == 1


def test_command_set_loss_samples_targets_that_exceed_slots():
    model = UnifiedModel(
        nn.Identity(),
        _FakeDecoder(),
        dual_decode=True,
        command_slot_decode=True,
        general_command_slots=2,
        user_specific_command_slots=2,
    )
    torch.manual_seed(7)
    _, sampled = model._slot_cost_matrices(
        torch.zeros(1, 2),
        [["first", "second", "third"]],
        head="general",
        num_slots=2,
    )
    assert len(sampled[0]) == 2
    assert set(sampled[0]).issubset({"first", "second", "third"})
