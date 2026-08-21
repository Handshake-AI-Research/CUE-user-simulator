"""Implicit-profile helpers for USP rollouts (no SFT dataset construction)."""

from __future__ import annotations

from pathlib import Path

from cue_training.baselines.common.jsonl import read_jsonl
from cue_training.baselines.common.llm import achat, parse_json_block
from cue_training.baselines.common.prompts import transcript_text
from cue_training.runlog.log import warn

PROFILES_FILE = "profiles.jsonl"

_PROFILE_PROMPT = (
    "Infer the IMPLICIT profile of the human user from the conversation below. "
    "Return STRICT JSON with keys: "
    '"goals" (what they want), "knowledge_level", "preferences", "personality", '
    '"communication_style". Keep each value to one short phrase.'
)


def profile_text(profile: dict) -> str:
    keys = ["goals", "knowledge_level", "preferences", "personality", "communication_style"]
    lines = [f"- {k.replace('_', ' ')}: {profile.get(k, 'unspecified')}" for k in keys]
    return "Implicit user profile:\n" + "\n".join(lines)


async def extract_profile_async(
    conversation: list[dict],
    *,
    model: str,
    api_key_env: str = "OPENAI_API_KEY",
    api_base: str | None = None,
) -> dict | None:
    """Extract one implicit profile (used by tau2/SimArena persona prep)."""

    transcript = transcript_text(conversation, max_turns=24)
    try:
        raw = await achat(
            [
                {"role": "system", "content": _PROFILE_PROMPT},
                {"role": "user", "content": f"Conversation:\n{transcript}"},
            ],
            model=model,
            api_key_env=api_key_env,
            api_base=api_base,
            temperature=0.3,
            max_tokens=400,
        )
    except Exception as exc:  # noqa: BLE001
        warn("usp", f"profile extraction failed: {exc}")
        return None
    parsed = parse_json_block(raw)
    return parsed if isinstance(parsed, dict) else None


def load_profiles(path: Path) -> list[dict]:
    if not path.exists():
        return []
    return read_jsonl(path)
