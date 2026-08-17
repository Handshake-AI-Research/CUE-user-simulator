"""Condition CUE on a dialogue, then steer an OpenAI-compatible simulator."""

from __future__ import annotations

import os
import sys

from cue import Cue, OpenAICompatSimulator
from cue.simulator import run_user_turn

CUE_REPO = os.environ.get("CUE_REPO", "handshake-ai-research/cue")
SIM_MODEL = os.environ.get("SIM_MODEL", "gpt-4o-mini")

conversation = [
    {"role": "user", "content": "can you tighten this paragraph"},
    {"role": "assistant", "content": "Sure — here is a shorter version."},
    {"role": "user", "content": "still too long, cut it in half"},
]

cue = Cue.from_pretrained(CUE_REPO, device=os.environ.get("CUE_DEVICE", "cpu"))
manual = cue.from_conversation(conversation)
sys.stdout.write(f"=== steering prompt ===\n{manual.steering_prompt}\n")

sim = OpenAICompatSimulator(SIM_MODEL)
scenario = "You want help rewriting a cover letter for a data analyst job."
user_msg = run_user_turn(sim, scenario=scenario, history=[], manual=manual)
sys.stdout.write(f"\n=== steered user (turn 1) ===\n{user_msg}\n")
