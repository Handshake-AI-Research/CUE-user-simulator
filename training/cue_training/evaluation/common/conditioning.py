"""Per-turn conditioning + history helpers shared across evaluation harnesses.

A ``TurnContext`` carries everything a user simulator needs for one episode: the
task goal (WHAT the user wants -- kept separate from HOW they behave), the CUE arm +
embedding (for the decoders), and an optional injected persona (for the baseline
matched/random ablation). History is always the neutral ``[{role, content}]`` shape
with ``user`` = our simulator's own past turns and ``assistant`` = the agent.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

Turn = dict[str, str]

# Arm names (CUE decoders vs baseline persona ablation vs floors).
CUE_ARMS = (
    "paired",
    "sample_shuffled",
    "dataset_mean",
    "sampled_uncond",
    "sampled_pop",
)
# Persona baselines reuse the CUE arm names: paired = the episode's own persona, sample_shuffled
# = a shuffled/other persona (analogous to the paired vs shuffled CUE embedding).
PERSONA_ARMS = ("paired", "sample_shuffled")
AS_IS = "as_is"


@dataclass
class TurnContext:
    """Per-episode conditioning handed to a user simulator."""

    variant: str  # decoder | baseline:<name> | base_api | base_local
    arm: str  # CUE: paired/sample_shuffled/dataset_mean | persona baseline: paired/sample_shuffled | as_is
    task: str = ""  # WHAT: the user's goal / scenario instructions
    domain: str | None = None
    episode_id: str | None = None
    # Harness-provided user-simulator system prompt (e.g. tau2's standard prompt), shared
    # across variants for comparability. When set it is the base system prompt for the
    # models we prompt directly, and is prepended to prompt-based baselines' own prompt.
    system_prompt: str | None = None
    cue_embedding: list[float] | None = None  # HOW (decoders)
    persona: Any = None  # HOW (baseline persona injection: paired/sample_shuffled); text or dict
    extra: dict[str, Any] = field(default_factory=dict)


def last_agent_text(history: list[Turn]) -> str:
    """Most recent assistant turn's text in a ``[{role, content}]`` history."""

    for turn in reversed(history or []):
        if str(turn.get("role")) == "assistant":
            return str(turn.get("content") or "").strip()
    return ""


def transcript_text(history: list[Turn], max_turns: int = 16) -> str:
    """Render a ``[{role, content}]`` history as a plain transcript."""

    if not history:
        return "(no messages yet)"
    lines = []
    for turn in history[-max_turns:]:
        role = "User" if str(turn.get("role")) == "user" else "Assistant"
        lines.append(f"{role}: {turn.get('content', '')}")
    return "\n".join(lines)
