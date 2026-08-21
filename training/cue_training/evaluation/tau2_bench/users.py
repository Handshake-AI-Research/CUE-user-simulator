"""tau2 HalfDuplexUser that delegates to our shared user simulators.

The class is built lazily (tau2 imported only when needed) so importing this module
does not require tau2. A single ``cue_eval_user`` handles every variant/arm by reading
its conditioning from ``llm_args`` (set per task by the runner). The tau2 conversation
(``state.messages``) is projected to the neutral ``[{role, content}]`` history our
sims expect (user = our sim, assistant = agent; agent-private tool messages skipped).
"""

from __future__ import annotations

from typing import Any

from cue_training.evaluation.common.conditioning import TurnContext
from cue_training.evaluation.common.user_sims import build_user_sim

USER_NAME = "cue_eval_user"


def _history_from_state(state: Any) -> list[dict[str, str]]:
    history: list[dict[str, str]] = []
    for msg in getattr(state, "messages", []) or []:
        role = str(getattr(msg, "role", "") or "").lower()
        content = getattr(msg, "content", None)
        is_tool_call = bool(getattr(msg, "tool_calls", None))
        if role == "user":
            if content:
                history.append({"role": "user", "content": str(content)})
        elif role == "assistant" and not is_tool_call and content:
            history.append({"role": "assistant", "content": str(content)})
        # tool / tool-call-only / system messages are agent-private -> skipped
    return history


def build_cue_eval_user_class() -> type:
    """Define + return the tau2 ``HalfDuplexUser`` subclass (imports tau2)."""

    from tau2.data_model.message import UserMessage  # type: ignore
    from tau2.data_model.persona import PersonaConfig  # type: ignore
    from tau2.user.user_simulator import (  # type: ignore
        SYSTEM_PROMPT,
        get_global_user_sim_guidelines,
    )
    from tau2.user.user_simulator_base import STOP, HalfDuplexUser, UserState  # type: ignore

    def _standard_system_prompt(instructions, tools) -> str:  # noqa: ANN001
        """tau2's standard user-simulator system prompt (guidelines + scenario).

        Byte-identical to ``UserSimulator.system_prompt`` so every model we prompt is
        framed exactly like tau2's native user simulator, without instantiating an LLM.
        """

        guidelines = get_global_user_sim_guidelines(use_tools=tools is not None)
        persona = PersonaConfig().to_guidelines_text() or ""
        if persona:
            persona = f"\n\n{persona}\n"
        guidelines = guidelines.replace("<PERSONA_GUIDELINES>", persona)
        return SYSTEM_PROMPT.format(
            global_user_sim_guidelines_with_persona=guidelines, instructions=instructions
        )

    class CueEvalUser(HalfDuplexUser):
        def __init__(self, instructions=None, tools=None, llm=None, llm_args=None):  # noqa: ANN001
            super().__init__(instructions=instructions, tools=tools)
            args = dict(llm_args or {})
            variant = str(args.get("variant", "base_api"))
            extra = dict(args.get("extra") or {})
            # Always: tau2 native user-sim system prompt (guidelines +
            # ``<scenario>{str(task.user_scenario)}</scenario>``) plus the stripped tau-usi
            # task-metadata block (``Task description:`` / ``Domain or topic:``) when available.
            # Decoder command blocks are appended later by DecoderUserSim. The tau2 agent + reward
            # are unaffected.
            from cue_training.evaluation.common.mirror_prompts import user_proxy_task_block

            scenario = extra.get("task_scenario")
            task = str(scenario or instructions or args.get("task") or "")
            system_prompt = _standard_system_prompt(instructions, tools)
            block = user_proxy_task_block(task_description=scenario, domain=args.get("domain"))
            if block:
                system_prompt = f"{system_prompt}\n\n{block}"
            self.ctx = TurnContext(
                variant=variant,
                arm=str(args.get("arm", "as_is")),
                task=task,
                domain=args.get("domain"),
                episode_id=args.get("episode_id"),
                cue_embedding=args.get("cue_embedding"),
                persona=args.get("persona"),
                system_prompt=system_prompt,
                extra=extra,
            )
            self.sim = build_user_sim(self.ctx)

        def set_seed(self, seed):  # noqa: ANN001, D401
            """No-op: the orchestrator calls set_seed; our sims manage their own sampling."""

        def get_init_state(self, message_history=None) -> UserState:  # noqa: ANN001
            return UserState(system_messages=[], messages=list(message_history or []))

        def generate_next_message(self, message, state):  # noqa: ANN001
            if message is not None and (getattr(message, "content", None) or getattr(message, "tool_calls", None)):
                state.messages.append(message)
            history = _history_from_state(state)
            text, done = self.sim.next_turn(history)
            if done and not text:
                text = STOP
            elif done:
                text = f"{text} {STOP}"
            # Persist any per-turn provenance (e.g. the hard decoder's injected command block)
            # on the message so it survives to the saved rollout via raw_data.
            cmd = getattr(self.sim, "last_command", None)
            raw_data = {"cue_command": cmd} if cmd else None
            user_message = UserMessage(role="user", content=text, raw_data=raw_data)
            state.messages.append(user_message)
            return user_message, state

    return CueEvalUser
