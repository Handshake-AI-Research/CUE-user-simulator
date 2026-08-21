"""Evolvable persona-policy generator (PPol).

A persona policy is parameterized by a *genome* (axis -> value). ``render_persona``
turns a genome into a prompt-augmentation string. The genome is the unit that the
evolutionary search in ``evolve.py`` mutates (locally and via an LLM program
mutation that appends free-text ``extra_rules``).
"""

from __future__ import annotations

import random

AXES: dict[str, list[str]] = {
    "verbosity": ["very terse", "terse", "moderate", "verbose"],
    "formality": ["very informal", "casual", "neutral", "formal"],
    "directness": ["blunt", "direct", "polite", "roundabout"],
    "expertise": ["novice", "intermediate", "expert"],
    "patience": ["impatient", "neutral", "patient"],
    "typo_rate": ["clean", "occasional typos", "frequent typos"],
    "emoji_use": ["none", "rare", "frequent"],
    "question_tendency": ["statements", "balanced", "many questions"],
}

_AXIS_HINT = {
    "verbosity": "Message length",
    "formality": "Register",
    "directness": "How you phrase requests",
    "expertise": "Domain skill level",
    "patience": "Tolerance for back-and-forth",
    "typo_rate": "Spelling/grammar care",
    "emoji_use": "Emoji usage",
    "question_tendency": "Statements vs questions",
}


def random_genome(rng: random.Random) -> dict:
    return {axis: rng.choice(values) for axis, values in AXES.items()}


def seed_genomes() -> list[dict]:
    rng = random.Random(0)
    seeds = [
        {axis: values[0] for axis, values in AXES.items()},
        {axis: values[-1] for axis, values in AXES.items()},
        {axis: values[len(values) // 2] for axis, values in AXES.items()},
    ]
    seeds += [random_genome(rng) for _ in range(3)]
    return seeds


def mutate_genome(genome: dict, rng: random.Random) -> dict:
    child = dict(genome)
    axis = rng.choice(list(AXES))
    child[axis] = rng.choice(AXES[axis])
    return child


def render_persona(genome: dict) -> str:
    lines = ["You communicate with these traits:"]
    for axis, value in genome.items():
        if axis == "extra_rules":
            continue
        hint = _AXIS_HINT.get(axis, axis)
        lines.append(f"- {hint}: {value}")
    extra = genome.get("extra_rules")
    if extra:
        lines.append("Additional behavioral rules:")
        for rule in extra if isinstance(extra, list) else [extra]:
            lines.append(f"- {rule}")
    return "\n".join(lines)
