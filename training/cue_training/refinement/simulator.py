"""Fixed LiteLLM/vLLM user simulator used by decoder refinement."""

from __future__ import annotations

import os
import re
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from typing import Any

from cue_training.preprocessing.llm import complete
from cue_training.utils.log import warn

_SYSTEM = (
    "You are simulating a real user in a conversation. Pursue the task goal while "
    "responding naturally to the assistant. Produce only the next USER message; "
    "never write an assistant reply, role label, analysis, or hidden reasoning."
)

# Surface form is the point of these examples: register (punctuation, casing, length) is
# what a sim2real discriminator reads, and the model's own polished default overrides an
# abstract rule, so the mechanics have to be named explicitly. But these are annotator
# sketches, not transcribed messages -- the annotation prompt asks for "short stylistic
# phrasing" with <ENTITY>/<NUM> placeholders and no task content. So point at the
# mechanics rather than the wording, and say what the placeholders are, or the model
# copies generated prose verbatim (which reads synthetic) or emits the brackets.
_EXAMPLES_HEADER = (
    "Style samples -- short sketches of how users like this one type. Match their "
    "mechanics, not their wording: punctuation (including missing, doubled, or absent "
    "end marks), capitalization, spacing, contractions, and how short the messages are. "
    "Angle-bracket slots like <ENTITY> or <NUM> stand in for details -- put your own "
    "specifics there and never write the brackets. Take nothing else from them, "
    "including their task content:\n"
)


def render_manual(manual: dict[str, Any] | None) -> str:
    """Render only simulator-safe commands and examples."""

    if not manual:
        return ""
    commands: list[str] = []
    examples: list[str] = []
    for entry in manual.get("commands") or []:
        if isinstance(entry, dict):
            text = str(entry.get("text") or entry.get("command") or "").strip()
            examples.extend(str(x).strip() for x in entry.get("examples") or [] if str(x).strip())
        else:
            text = str(entry).strip()
        if text:
            commands.append(text)
    examples.extend(str(x).strip() for x in manual.get("examples") or [] if str(x).strip())
    sections = []
    if commands:
        sections.append("Behavior manual:\n" + "\n".join(f"- {text}" for text in commands))
    if examples:
        sections.append(_EXAMPLES_HEADER + "\n".join(
            f"- {text}" for text in examples
        ))
    return "\n\n".join(sections)


def render_dual_manual(
    general: dict[str, Any] | None,
    specific: dict[str, Any] | None,
    style: dict[str, Any] | None = None,
) -> str:
    """Render general + user-specific (+ optional style) sections for steering."""

    sections: list[str] = []
    gcmds = [str(c).strip() for c in (general or {}).get("commands") or [] if str(c).strip()]
    scmds = [str(c).strip() for c in (specific or {}).get("commands") or [] if str(c).strip()]
    stcmds = [str(c).strip() for c in (style or {}).get("commands") or [] if str(c).strip()]
    examples = [
        str(x).strip()
        for x in (
            list((specific or {}).get("examples") or [])
            + list((style or {}).get("examples") or [])
        )
        if str(x).strip()
    ]
    if gcmds:
        sections.append("General behavior:\n" + "\n".join(f"- {text}" for text in gcmds))
    if scmds:
        sections.append("User-specific behavior:\n" + "\n".join(f"- {text}" for text in scmds))
    if stcmds:
        sections.append("Writing style:\n" + "\n".join(f"- {text}" for text in stcmds))
    if examples:
        sections.append(_EXAMPLES_HEADER + "\n".join(f"- {text}" for text in examples))
    return "\n\n".join(sections)


def build_messages(
    history: list[dict[str, Any]],
    *,
    task_text: str,
    manual: dict[str, Any] | None = None,
    system_instruction: str = _SYSTEM,
) -> list[dict[str, str]]:
    """Build role-compatible chat messages for a next-user completion."""

    system = (
        f"{system_instruction.strip() or _SYSTEM}\n\n"
        f"Task goal:\n{task_text.strip() or 'Continue the conversation.'}"
    )
    rendered = render_manual(manual)
    if rendered:
        system += f"\n\n{rendered}"
    messages = [{"role": "system", "content": system}]
    for turn in history:
        role = str(turn.get("role") or "").lower()
        if role == "system":
            role = "assistant"
        if role not in {"user", "assistant"}:
            continue
        content = str(turn.get("content") or "").strip()
        if content:
            messages.append({"role": role, "content": content})
    # Gemini rejects conversations that end on a model/assistant turn. decoder refinement
    # histories are prefixes before the next user message, so they usually end
    # assistant-side; cue the next user turn the same way data annotation does.
    if not messages or messages[-1]["role"] != "user":
        messages.append(
            {
                "role": "user",
                "content": "Continue as the user with your next message only.",
            }
        )
    return messages


def clean_user_message(text: str) -> str:
    """Remove common reasoning blocks and generated role wrappers."""

    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL | re.IGNORECASE)
    text = re.sub(r"```(?:text)?\s*", "", text, flags=re.IGNORECASE).replace("```", "")
    text = re.sub(r"^\s*(?:next\s+)?(?:user|human)\s*(?:message)?\s*:\s*", "", text, flags=re.IGNORECASE)
    text = re.split(r"\n\s*(?:assistant|system)\s*:", text, maxsplit=1, flags=re.IGNORECASE)[0]
    return text.strip().strip("\"'")


@dataclass(frozen=True)
class UserSimulator:
    model: str
    api_base: str | None = None
    api_key: str | None = None
    api_key_env: str | None = None
    temperature: float = 0.7
    max_tokens: int = 128
    max_input_tokens: int | None = None
    system_instruction: str = _SYSTEM

    def simulate(
        self,
        *,
        history: list[dict[str, Any]],
        task_text: str,
        manual: dict[str, Any] | None = None,
    ) -> str:
        """Generate one fixed-model user turn; omit manual for the base rollout."""

        api_key = self.api_key or (os.environ.get(self.api_key_env) if self.api_key_env else None)
        try:
            text = complete(
                model=self.model,
                messages=build_messages(
                    history,
                    task_text=task_text,
                    manual=manual,
                    system_instruction=self.system_instruction,
                ),
                api_base=self.api_base,
                api_key=api_key,
                temperature=self.temperature,
                max_tokens=self.max_tokens,
                max_input_tokens=self.max_input_tokens,
            )
            return clean_user_message(text)
        except Exception as exc:  # noqa: BLE001 - one failed rollout should not abort RL
            warn("refinement:sim", f"{self.model} rollout failed: {exc}")
            return ""

    def simulate_base(self, *, history: list[dict[str, Any]], task_text: str) -> str:
        return self.simulate(history=history, task_text=task_text, manual=None)

    def simulate_batch(
        self,
        requests: list[dict[str, Any]],
        *,
        max_workers: int = 8,
    ) -> list[str]:
        """Generate requests concurrently while preserving input order."""

        with ThreadPoolExecutor(max_workers=max(1, max_workers)) as pool:
            return list(pool.map(lambda request: self.simulate(**request), requests))
