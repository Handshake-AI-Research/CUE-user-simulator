"""Periodic discriminative Turing-judge audits for decoder refinement."""

from __future__ import annotations

import hashlib
import os
import re
from dataclasses import dataclass
from typing import Any

from cue_training.preprocessing.llm import complete


def normalize_score(score: float) -> float:
    """Turing reward: cap the candidate-oriented 1--7 score at 5."""

    return (min(5.0, max(1.0, float(score))) - 1.0) / 6.0


def length_penalty(human_turn: str, simulator_turn: str, *, weight: float = 0.25) -> float:
    """Penalize large relative word-count differences."""

    human_words = max(1, len(human_turn.split()))
    simulator_words = len(simulator_turn.split())
    return min(0.4, max(0.0, weight * abs(simulator_words - human_words) / human_words))


def parse_judge_score(text: str) -> int:
    """Parse one pairwise 1--7 Turing score."""

    match = re.search(r"-?\d+", text)
    if not match:
        raise ValueError("judge response has no integer")
    return min(7, max(1, int(match.group(0))))


@dataclass(frozen=True)
class Judge:
    model: str
    api_base: str | None = None
    api_key: str | None = None
    api_key_env: str | None = None


class RewardPanel:
    """Average available judge ratings on configured periodic audit steps."""

    def __init__(
        self,
        judges: list[Judge],
        *,
        audit_interval: int = 100,
        seed: int = 0,
        length_penalty_weight: float = 0.25,
        max_tokens: int = 128,
    ) -> None:
        self.judges = list(judges)
        self.audit_interval = max(1, audit_interval)
        self.seed = seed
        self.length_penalty_weight = max(0.0, length_penalty_weight)
        self.max_tokens = max_tokens

    def should_audit(self, step: int) -> bool:
        return step >= 0 and step % self.audit_interval == 0

    def _simulator_is_a(self, *, step: int, session_id: str, model: str) -> bool:
        digest = hashlib.sha256(
            f"{self.seed}:{step}:{session_id}:{model}".encode()
        ).digest()
        return bool(digest[0] & 1)

    def audit(
        self,
        *,
        step: int,
        session_id: str,
        history: list[dict[str, Any]],
        human_turn: str,
        simulator_turn: str,
        force: bool = False,
    ) -> dict[str, Any] | None:
        """Run the panel only on audit steps; failed judges are omitted."""

        if not force and not self.should_audit(step):
            return None
        transcript = "\n".join(
            f"{str(turn.get('role') or '').upper()}: {str(turn.get('content') or '')}"
            for turn in history
        )
        scores: list[float] = []
        errors: dict[str, str] = {}
        for judge in self.judges:
            simulator_is_a = self._simulator_is_a(
                step=step, session_id=session_id, model=judge.model
            )
            a, b = (
                (simulator_turn, human_turn) if simulator_is_a else (human_turn, simulator_turn)
            )
            messages = [
                {
                    "role": "system",
                    "content": (
                        "You are a Turing-test judge for a user simulator. Given the conversation "
                        "context and TWO candidate next USER messages, decide which one was written "
                        "by the REAL human user. Reply with ONLY one integer from 1 to 7: "
                        "1 = Response A is almost certainly real, 4 = equally likely, "
                        "7 = Response B is almost certainly real."
                    ),
                },
                {
                    "role": "user",
                    "content": (
                        f"CONTEXT:\n{transcript}\n\nResponse A:\n{a}\n\nResponse B:\n{b}\n\n"
                        "Which response was written by the real human? Reply with one integer 1-7."
                    ),
                },
            ]
            try:
                raw = complete(
                    model=judge.model,
                    messages=messages,
                    temperature=0.0,
                    max_tokens=min(self.max_tokens, 8),
                    api_base=judge.api_base,
                    api_key=judge.api_key
                    or (os.environ.get(judge.api_key_env) if judge.api_key_env else None),
                )
                raw_score = parse_judge_score(raw)
                candidate_score = 8 - raw_score if simulator_is_a else raw_score
                scores.append(normalize_score(candidate_score))
            except Exception as exc:  # noqa: BLE001 - unavailable judges are intentionally skipped
                errors[judge.model] = str(exc)
        if not scores:
            return {
                "score": None,
                "judge_scores": [],
                "length_penalty": 0.0,
                "errors": errors,
            }
        penalty = length_penalty(
            human_turn, simulator_turn, weight=self.length_penalty_weight
        )
        return {
            "score": max(0.0, min(2.0 / 3.0, sum(scores) / len(scores) - penalty)),
            "judge_scores": scores,
            "length_penalty": penalty,
            "errors": errors,
        }
