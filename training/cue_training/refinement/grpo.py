"""Clipped GRPO objective for persona-manual continuations."""

from __future__ import annotations

from typing import Any

import torch


def group_advantages(
    rewards: torch.Tensor,
    *,
    num_candidates: int,
    eps: float = 1e-6,
) -> torch.Tensor:
    if rewards.numel() % num_candidates:
        raise ValueError("reward count must be divisible by num_candidates")
    grouped = rewards.view(-1, num_candidates)
    return ((grouped - grouped.mean(dim=1, keepdim=True)) / (
        grouped.std(dim=1, keepdim=True, unbiased=False) + eps
    )).reshape(-1)


def clipped_grpo_loss(
    decoder: Any,
    reference_decoder: Any,
    *,
    bottleneck: torch.Tensor,
    sentinel_ids: torch.Tensor,
    token_ids: torch.Tensor,
    completion_mask: torch.Tensor,
    old_token_logprobs: torch.Tensor,
    rewards: torch.Tensor,
    num_candidates: int,
    segments: list[tuple[int, int, int]] | None = None,
    clip_eps: float = 0.2,
    kl_beta: float = 1e-3,
) -> tuple[torch.Tensor, dict[str, float]]:
    """Token-masked GRPO/PPO ratio objective with sampled-token reference KL."""

    current = decoder.manual_token_logprobs(
        bottleneck=bottleneck,
        sentinel_ids=sentinel_ids,
        token_ids=token_ids,
        completion_mask=completion_mask,
        segments=segments,
    )
    with torch.no_grad():
        reference = reference_decoder.manual_token_logprobs(
            bottleneck=bottleneck,
            sentinel_ids=sentinel_ids,
            token_ids=token_ids,
            completion_mask=completion_mask,
            segments=segments,
        )
    advantages = group_advantages(
        rewards.to(current.device), num_candidates=num_candidates
    ).unsqueeze(-1)
    ratio = torch.exp(
        (current - old_token_logprobs.to(current.device)).clamp(min=-20.0, max=20.0)
    )
    unclipped = ratio * advantages
    clipped = ratio.clamp(1.0 - clip_eps, 1.0 + clip_eps) * advantages
    mask = completion_mask.to(current.dtype)
    denom = mask.sum().clamp(min=1.0)
    policy_loss = -(torch.minimum(unclipped, clipped) * mask).sum() / denom

    # Positive sampled-token KL estimator used by GRPO implementations.
    log_ratio_ref = (reference - current).clamp(min=-20.0, max=20.0)
    per_token_kl = torch.exp(log_ratio_ref) - log_ratio_ref - 1.0
    kl = (per_token_kl * mask).sum() / denom
    loss = policy_loss + kl_beta * kl
    clip_fraction = (
        ((ratio - 1.0).abs() > clip_eps).to(mask.dtype) * mask
    ).sum() / denom
    return loss, {
        "policy_loss": float(policy_loss.detach()),
        "kl": float(kl.detach()),
        "clip_fraction": float(clip_fraction.detach()),
        "reward_mean": float(rewards.mean().detach()),
        "reward_std": float(rewards.std(unbiased=False).detach()),
    }
