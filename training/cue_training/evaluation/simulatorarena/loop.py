"""Reimplemented document-creation conversation loop.

Substitutes our user-turn generator (evaluation.common.user_sims) for SimulatorArena's
prompt-driven user, while the assistant turn comes from the shared assistant client
(bridge.assistant_reply). Produces a plain user/assistant conversation the runner then
converts to SimulatorArena-format output and MirrorBench rollouts.
"""

from __future__ import annotations

from typing import Any

from cue_training.evaluation.common.user_sims import UserSim
from cue_training.evaluation.simulatorarena import bridge
from cue_training.runlog.log import warn

_ASSISTANT_SYSTEM = (
    "You are a skilled writing assistant. Your role is to help users create and edit "
    "documents that should be under 600 words by following their specific instructions "
    "and requirements."
)

_MATH_ASSISTANT_SYSTEM = (
    "You are a skilled math tutor. Your goal is to help students understand and solve "
    "problems independently. Provide guidance based on their questions or mistakes. Ask "
    "questions to encourage their thinking and let students do most of the work themselves. "
    "Never give out the solution directly to students."
)

_ASSISTANT_SYSTEM_BY_DOMAIN = {"writing": _ASSISTANT_SYSTEM, "math": _MATH_ASSISTANT_SYSTEM}


def simulate_case(
    *,
    user_sim: UserSim,
    assistant_model: str,
    max_turns: int = 12,
    assistant_max_tokens: int = 1024,
    domain: str = "writing",
    task: str = "",
    assistant_router: Any = None,
) -> list[dict[str, Any]]:
    assistant_system = _ASSISTANT_SYSTEM_BY_DOMAIN.get(domain, _ASSISTANT_SYSTEM)
    conversation: list[dict[str, Any]] = []
    user_turns = 0
    while user_turns < max_turns:
        text, done = user_sim.next_turn(conversation)
        text = (text or "").strip()
        if not text:
            break
        # Match tau2: persist decoder provenance on the turn so MirrorBench rollouts keep
        # ``metadata.cue_command`` (command_block / decoded) for paired/sampled inspection.
        user_turn: dict[str, Any] = {"role": "user", "content": text}
        cmd = getattr(user_sim, "last_command", None)
        if cmd:
            user_turn["raw_data"] = {"cue_command": cmd}
        conversation.append(user_turn)
        user_turns += 1
        if done:
            break
        assistant_conversation = conversation
        if domain == "math" and conversation:
            # Native SimulatorArena tells the tutor the problem on its first user turn,
            # while preserving only the student's generated query in the saved conversation.
            assistant_conversation = [
                {
                    "role": "user",
                    "content": (
                        f"Here is the problem that you will tutor me on:\n{task.strip()}\n\n"
                        f"{conversation[0]['content']}"
                    ),
                },
                *conversation[1:],
            ]
        messages = [{"role": "system", "content": assistant_system}, *assistant_conversation]
        model = assistant_router.route(assistant_model) if assistant_router else assistant_model
        try:
            reply = bridge.assistant_reply(model, messages, max_tokens=assistant_max_tokens)
        except Exception as exc:  # noqa: BLE001
            alt = assistant_router.mark_dead(model) if assistant_router else None
            if alt is None:
                warn("simarena", f"assistant turn failed: {exc}")
                break
            try:
                reply = bridge.assistant_reply(alt, messages, max_tokens=assistant_max_tokens)
            except Exception as exc2:  # noqa: BLE001
                warn("simarena", f"assistant turn failed (fallback {alt}): {exc2}")
                break
        conversation.append({"role": "assistant", "content": (reply or "").strip()})
    return conversation
