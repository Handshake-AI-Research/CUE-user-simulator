"""Generic closed-loop conversation driver for normalized MirrorBench datasets."""

from __future__ import annotations

from typing import Any, Callable

from cue_training.evaluation.common.conditioning import TurnContext
from cue_training.evaluation.common.mirrorbench_export import to_mirrorbench_rollout
from cue_training.evaluation.common.mirror_prompts import user_proxy_system_prompt
from cue_training.evaluation.common.user_sims import UserSim, build_user_sim
from cue_training.runlog.log import warn

Turn = dict[str, str]

AssistantFn = Callable[[list[Turn], str], str]


def count_user_turns(real_conversation: list[dict[str, Any]]) -> int:
    return sum(1 for t in real_conversation or [] if str(t.get("role")) == "user")


def opening_prompt(row: dict[str, Any]) -> str | None:
    """The human's first user message, which sets the topic of a PRISM conversation."""

    for turn in row.get("real_conversation") or []:
        if str(turn.get("role")) == "user":
            return str(turn.get("content") or "").strip() or None
    return None


def prism_user_system_prompt(row: dict[str, Any]) -> str:
    """Build PRISM continuation instructions while leaving persona injection additive."""

    metadata = row.get("metadata") or {}
    task = str(row.get("task_description") or row.get("intent") or "")
    domain = str(metadata.get("domain") or "diversity")
    raw_type = str(
        metadata.get("conversation_type")
        or metadata.get("conversation_type_label")
        or row.get("conversation_type")
        or ""
    ).strip()
    normalized = raw_type.lower().replace("_", " ").replace("-", " ")
    type_guidance = {
        "unguided": (
            "Unguided: the participant chose the topic and opening without a prescribed "
            "value or controversy."
        ),
        "values guided": (
            "Values-guided: continue exploring the values already established by the user; "
            "do not invent additional beliefs."
        ),
        "controversy guided": (
            "Controversy-guided: continue the controversial topic from the user's established "
            "perspective without artificially escalating it."
        ),
    }.get(normalized)

    blocks = [
        user_proxy_system_prompt(task_description=task, domain=domain),
        (
            "## PRISM conversation continuation\n"
            "The real human's opening message is already the first user turn in the conversation "
            "history. Continue that same conversation. Do not restart or restate the opening, "
            "introduce a replacement scenario, or answer as the assistant. Treat the conversation "
            "history as authoritative."
        ),
    ]
    if type_guidance:
        blocks.append(f"## PRISM conversation condition\n{type_guidance}")
    blocks.append(
        "## Persona and behavioral conditioning\n"
        "Any persona, profile, or behavioral manual supplied below controls how this specific "
        "user communicates. It must not replace the established topic, goal, or conversation history."
    )
    return "\n\n".join(blocks)


def default_turn_budget(row: dict[str, Any], *, cap: int | None = None) -> int:
    n = int((row.get("metadata") or {}).get("n_user_turns") or 0)
    if n <= 0:
        n = count_user_turns(list(row.get("real_conversation") or []))
    n = max(1, n)
    if cap is not None:
        n = min(n, int(cap))
    return n


def run_closed_loop(
    *,
    row: dict[str, Any],
    user_sim: UserSim,
    assistant_reply: AssistantFn,
    assistant_system: str,
    max_user_turns: int,
    provenance: dict[str, Any],
    opening: str | None = None,
) -> dict[str, Any]:
    conversation: list[Turn] = []
    user_turns = 0
    # Seed the real opening prompt verbatim (it determines the conversation); the simulator
    # then drives every subsequent user turn. Like every user turn, it gets an assistant reply.
    if opening:
        conversation.append({"role": "user", "content": opening})
        user_turns += 1
        try:
            reply = assistant_reply(conversation, assistant_system)
            conversation.append({"role": "assistant", "content": (reply or "").strip()})
        except Exception as exc:  # noqa: BLE001
            warn("closed_loop", f"assistant turn failed: {exc}")
            max_user_turns = user_turns  # stop before the sim loop
    while user_turns < max_user_turns:
        text, done = user_sim.next_turn(conversation)
        text = (text or "").strip()
        if not text:
            break
        conversation.append({"role": "user", "content": text})
        user_turns += 1
        if done:
            break
        try:
            reply = assistant_reply(conversation, assistant_system)
        except Exception as exc:  # noqa: BLE001
            warn("closed_loop", f"assistant turn failed: {exc}")
            break
        conversation.append({"role": "assistant", "content": (reply or "").strip()})
    return to_mirrorbench_rollout(
        source_record=row,
        conversation=conversation,
        provenance=provenance,
    )


def build_episode_sim(
    *,
    variant: str,
    arm: str,
    row: dict[str, Any],
    persona: Any = None,
    cue_embedding: list[float] | None = None,
    extra: dict[str, Any] | None = None,
    system_prompt: str | None = None,
) -> UserSim:
    task = str(row.get("task_description") or row.get("intent") or "")
    domain = str((row.get("metadata") or {}).get("domain") or row.get("dataset") or "diversity")
    ctx = TurnContext(
        variant=variant,
        arm=arm,
        task=task,
        domain=domain,
        episode_id=str(row.get("episode_id") or row.get("task_id") or ""),
        cue_embedding=cue_embedding,
        persona=persona,
        system_prompt=system_prompt,
        extra=extra or {},
    )
    return build_user_sim(ctx)
