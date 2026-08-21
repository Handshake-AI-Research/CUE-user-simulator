"""Encoder, decoder, and loss internals: system fusion, flamingo gates, truncation, InfoNCE."""

from __future__ import annotations

import torch

from cue_training.decoder.flamingo import GatedXAttnDense
from cue_training.encoder.system_fusion import SystemContextFusion
from cue_training.model import command_overlap_infonce, greedy_command_overlap


def test_fusion_is_query_dependent():
    fusion = SystemContextFusion(hidden_dim=32, num_heads=4)
    system = torch.randn(2, 4, 32)
    mask = torch.ones(2, 4, dtype=torch.bool)
    has = torch.tensor([True, True])
    q1 = torch.randn(2, 32)
    q2 = torch.randn(2, 32)
    o1 = fusion(q1, system, mask, has)
    o2 = fusion(q2, system, mask, has)
    assert not torch.allclose(o1, o2, atol=1e-5)


def test_no_system_passthrough():
    fusion = SystemContextFusion(hidden_dim=32, num_heads=4)
    q = torch.randn(2, 32)
    has = torch.tensor([False, False])
    out = fusion(q, None, None, has)
    assert torch.allclose(out, q)


def test_zero_init_gates_are_identity():
    block = GatedXAttnDense(hidden_dim=32, num_heads=4)
    hidden = torch.randn(2, 5, 32)
    persona = torch.randn(2, 3, 32)
    out = block(hidden, persona)
    # With alpha=0, tanh(alpha)=0 so residual additions are zero.
    assert torch.allclose(out, hidden, atol=1e-5)


def test_nonzero_gates_change_hidden():
    block = GatedXAttnDense(hidden_dim=32, num_heads=4)
    with torch.no_grad():
        block.alpha_xattn.fill_(2.0)
        block.alpha_dense.fill_(2.0)
    hidden = torch.randn(2, 5, 32)
    persona = torch.randn(2, 3, 32)
    out = block(hidden, persona)
    assert not torch.allclose(out, hidden, atol=1e-4)


class _FakeTok:
    """Whitespace tokenizer stand-in (1 token per whitespace-separated piece)."""

    def encode(self, text: str, add_special_tokens: bool = False, verbose: bool = True) -> list[int]:  # noqa: ARG002
        parts = text.split()
        return list(range(len(parts)))

    def decode(self, ids: list[int], skip_special_tokens: bool = True) -> str:  # noqa: ARG002
        # Round-trip identity isn't needed; return a marker with lengths.
        return f"<{len(ids)}tok>"


def test_truncate_text_head_tail_noop_when_short() -> None:
    from cue_training.encoder.truncate import truncate_text_head_tail

    tok = _FakeTok()
    text = "hello world"
    assert truncate_text_head_tail(text, tok, max_length=256) == text


def test_truncate_text_head_tail_keeps_both_ends() -> None:
    from cue_training.encoder.truncate import truncate_text_head_tail

    tok = _FakeTok()
    # Body alone is 400 tokens; with "passage:" (1 token) exceeds max_length=256.
    words = [f"w{i}" for i in range(400)]
    text = " ".join(words)
    out = truncate_text_head_tail(
        text, tok, max_length=256, head_tokens=128, tail_tokens=128, prefix="passage: "
    )
    # Fake decode returns length marker; budget = 256 - 1 = 255 → head 128 + tail 127.
    assert out.startswith("<")
    n = int(out.strip("<>tok"))
    assert n == 255


def test_greedy_command_overlap_is_symmetric_and_matches_identical_sets():
    commands = torch.tensor(
        [
            [[1.0, 0.0], [0.0, 1.0]],
            [[1.0, 0.0], [-1.0, 0.0]],
        ]
    )
    mask = torch.ones(2, 2, dtype=torch.bool)
    overlap = greedy_command_overlap(commands, mask, commands, mask)
    assert torch.allclose(overlap, overlap.T)
    assert torch.allclose(overlap.diag(), torch.ones(2))


def test_command_overlap_infonce_penalizes_collapsed_cue_vectors():
    overlap = torch.eye(2)
    aligned = torch.eye(2)
    collapsed = torch.tensor([[1.0, 0.0], [1.0, 0.0]])
    aligned_loss = command_overlap_infonce(
        aligned, aligned, overlap, temperature=0.1, target_temperature=0.1
    )
    collapsed_loss = command_overlap_infonce(
        collapsed, collapsed, overlap, temperature=0.1, target_temperature=0.1
    )
    assert aligned_loss < collapsed_loss
