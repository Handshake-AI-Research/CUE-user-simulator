"""Persona manual returned by CUE encode/decode or sampling."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal


@dataclass
class PersonaManual:
    """Decoded behavioral commands plus a ready-to-inject steering prompt.

    ``steering_prompt`` is what you append to a user-simulator system message.
    ``commands`` / ``examples`` are the flat lists; ``general`` / ``specific`` /
    ``style`` keep the three decode heads when dual decode is on.
    """

    commands: list[str]
    examples: list[str] = field(default_factory=list)
    steering_prompt: str = ""
    general: list[str] = field(default_factory=list)
    specific: list[str] = field(default_factory=list)
    style: list[str] = field(default_factory=list)
    embedding: list[float] | None = None
    source: Literal["conditioned", "sampled"] = "conditioned"
    raw: dict[str, Any] | None = None

    def as_system_addon(self, *, prefix: str = "Behave like this specific user:\n") -> str:
        """Text to append under a scenario system prompt."""

        body = (self.steering_prompt or "").strip()
        if not body:
            return ""
        return f"{prefix}{body}"

    def to_dict(self, *, include_embedding: bool = False) -> dict[str, Any]:
        """Return the stable, JSON-serializable application contract."""

        payload: dict[str, Any] = {
            "source": self.source,
            "commands": self.commands,
            "examples": self.examples,
            "general": self.general,
            "specific": self.specific,
            "style": self.style,
            "steering_prompt": self.steering_prompt,
        }
        if include_embedding:
            payload["embedding"] = self.embedding
        return payload
