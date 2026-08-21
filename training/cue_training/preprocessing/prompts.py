"""Prompts for dual-contrast persona manual extraction."""

from __future__ import annotations

from typing import Any

EXTRACTION_SYSTEM = """You write role-playable persona manuals for user simulators.
Output ONLY valid JSON with this schema:
{
  "schema_version": 1,
  "commands": [
    {
      "text": "imperative behavioral/stylistic rule",
      "kind": "sim_contrast" | "human_contrast",
      "form": "general" | "if_then",
      "examples": ["short stylistic phrasing with placeholders if needed"],
      "evidence_turn_ids": ["turn_id from TARGET only", "..."]
    }
  ]
}
Hard rules:
- Exactly 10 commands: first 5 sim_contrast, next 5 human_contrast.
- Every command MUST include evidence_turn_ids citing TARGET turn_id values that support it.
- Write BEHAVIOR/STYLE rules only. Ban task content: product names, order/account IDs,
  passwords/PINs, membership tiers, refund/return policy facts, addresses, emails, prices.
- Prefer placeholders like <ENTITY>, <NUM>, <TERM> in examples; never copy rare nouns from the task.
- sim_contrast: what separates this human from naive LLM user sims (generic humanness / anti-assistant tells).
- human_contrast: what separates THIS session from the human negatives — concrete, session-specific
  habits (timing, order of answers, digressions, pushback, compliance pattern). Do NOT use vague
  filler like "be polite/cooperative/calm/matter-of-fact".
- At least 2 human_contrast commands MUST encode habits that are true of TARGET and NOT also true
  of the human negatives (e.g. answer lag, escalation, pushback, impatience, try-then-report,
  smalltalk digression, playful tests, acceptance-after-refusal). Soft closers/thanks shared by
  most humans do not count toward those 2.
- Turn-level behavior is critical. Cover what shows up in the TARGET turns:
  1) Timing: if the user answers a PREVIOUS assistant question (answer lag / out-of-order replies),
     include at least one human_contrast if_then about that lag, with evidence on the lagged turns.
     Do NOT claim the user answers the current/latest prompt in order when the transcript shows lag.
  2) Reaction behavior: encode how the user reacts to assistant moves with if_then human_contrast
     commands when present — e.g. acceptance vs pushback after a refusal/policy block, escalation
     requests, impatience/hurry, brief gratitude, try-then-report after a suggested fix, quick close
     after resolution, clarification after a wrong assumption. Cite the reacting user turn_ids.
     Only use reactions the transcript actually shows; never invent frustration, anger, or warmth.
- Commands must be TRUE of the TARGET transcript. Do not invent emotions or reactions the user never shows.
- Prefer if_then forms for human_contrast when the transcript shows a reactive pattern.
- Keep examples short and stylistic (not content spoilers).
"""


def format_turns(turns: list[dict[str, Any]], *, max_chars: int = 400, max_turns: int = 0) -> str:
    selected = turns[-max_turns:] if max_turns and len(turns) > max_turns else turns
    lines = []
    for turn in selected:
        role = turn.get("role", "")
        tid = turn.get("turn_id") or ""
        content = str(turn.get("content") or "")
        if len(content) > max_chars:
            content = content[: max_chars - 1] + "…"
        prefix = f"{role}[{tid}]" if tid else role
        lines.append(f"{prefix}: {content}")
    return "\n".join(lines)


def build_extraction_messages(
    *,
    target_turns: list[dict[str, Any]],
    human_negatives: list[list[dict[str, Any]]],
    simulator_turns: dict[str, list[str]],
) -> list[dict[str, str]]:
    neg_blocks = []
    for i, neg in enumerate(human_negatives, start=1):
        neg_blocks.append(f"[Human negative {i}]\n{format_turns(neg, max_chars=220, max_turns=8)}")
    sim_blocks = []
    for model, responses in simulator_turns.items():
        joined = "\n".join(f"- {str(r)[:220]}" for r in responses[:15] if str(r).strip())
        sim_blocks.append(f"[Simulator {model} user responses]\n{joined}")
    user_ids = [str(t.get("turn_id")) for t in target_turns if t.get("role") == "user" and t.get("turn_id")]
    user = (
        "TARGET HUMAN SESSION (cite these turn_ids in evidence_turn_ids):\n"
        f"user_turn_ids={user_ids}\n"
        f"{format_turns(target_turns, max_chars=350)}\n\n"
        + "\n\n".join(neg_blocks)
        + "\n\n"
        + "\n\n".join(sim_blocks)
        + "\n\nWrite one 10-command persona manual as a single JSON object. "
        "Every command needs evidence_turn_ids from the TARGET list above. "
        "Include turn-level timing (lag/out-of-order) and user reaction behaviors when they appear. "
        "At least 2 human_contrast commands must differentiate TARGET from the human negatives."
    )
    return [
        {"role": "system", "content": EXTRACTION_SYSTEM},
        {"role": "user", "content": user},
    ]
