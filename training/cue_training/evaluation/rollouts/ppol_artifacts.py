"""PPOL artifact + persona-pool helpers: train once per sim on tau2, reuse elsewhere."""

from __future__ import annotations

import random
from pathlib import Path
from typing import Any

from cue_training.evaluation.common.io import read_jsonl, write_jsonl
from cue_training.evaluation.rollouts.config import SimulatorConfig


PPOL_TRAIN_DOMAIN = "retail_airline"
# Paper final N after curriculum 5→8→10.
PPOL_N_PERSONAS = 10
# Paper generator / mutator (LiteLLM).
PPOL_GEN_MODEL = "openrouter/google/gemini-3-flash-preview"


def ppol_sim_model(sim: SimulatorConfig) -> str:
    """LiteLLM id used for PPOL τ² fitness user-sim for this simulator."""

    if sim.kind == "api":
        return sim.model
    return f"hosted_vllm/{sim.served_name or sim.model}"


def ppol_gen_model(paper_overrides: dict[str, Any] | None = None) -> str:
    """LiteLLM id for G(c,D,N) / OpenEvolve mutator (decoupled from the fitness user-sim)."""

    ov = dict(paper_overrides or {})
    return str(ov.get("gen_model") or ov.get("mutation_model") or PPOL_GEN_MODEL)


def ppol_n_personas(paper_overrides: dict[str, Any] | None = None) -> int:
    ov = dict(paper_overrides or {})
    raw = ov.get("n_personas", ov.get("n", PPOL_N_PERSONAS))
    try:
        return max(1, int(raw))
    except (TypeError, ValueError):
        return PPOL_N_PERSONAS


def ppol_simtag(sim_model: str) -> str:
    return str(sim_model).replace("/", "__").replace(":", "__")


def ppol_artifacts_dir(
    artifact_root: Path,
    sim_model: str,
    *,
    domain: str = PPOL_TRAIN_DOMAIN,
) -> Path:
    return Path(artifact_root) / "ppol" / f"{domain}_{ppol_simtag(sim_model)}"


def ppol_best_program_path(
    artifact_root: Path,
    sim_model: str,
    *,
    domain: str = PPOL_TRAIN_DOMAIN,
    override: Path | None = None,
) -> Path:
    if override is not None:
        return Path(override)
    return ppol_artifacts_dir(artifact_root, sim_model, domain=domain) / "best_program.py"


def persona_pool_path(artifacts_dir: Path) -> Path:
    return Path(artifacts_dir) / "personas_pool.jsonl"


def tau2_personas_path(artifacts_dir: Path) -> Path:
    return Path(artifacts_dir) / "personas_tau2.jsonl"


def load_persona_pool(path: Path | str) -> list[str]:
    """Load persona strings from a pool or a full sidecar JSONL."""

    out: list[str] = []
    seen: set[str] = set()
    for row in read_jsonl(str(path)):
        text = row.get("persona")
        if not isinstance(text, str) or not text.strip():
            continue
        if text in seen:
            continue
        seen.add(text)
        out.append(text)
    return out


def write_persona_pool(path: Path | str, personas: list[str]) -> None:
    rows = [{"persona": p} for p in personas if isinstance(p, str) and p.strip()]
    write_jsonl(str(path), rows)


def update_persona_pool_from_sidecar(pool_path: Path, sidecar_path: Path) -> list[str]:
    """Merge persona texts from a sidecar into the shared pool; return the full pool."""

    existing = load_persona_pool(pool_path) if Path(pool_path).is_file() else []
    merged = load_persona_pool(sidecar_path)
    seen = set(existing)
    for p in merged:
        if p not in seen:
            existing.append(p)
            seen.add(p)
    write_persona_pool(pool_path, existing)
    return existing


def sidecar_covers_as_is(path: Path | str, variant: str) -> bool:
    """True if ``path`` has at least one ``as_is`` row for ``variant``."""

    p = Path(path)
    if not p.is_file() or p.stat().st_size == 0:
        return False
    for row in read_jsonl(str(p)):
        if row.get("variant") == variant and row.get("arm") == "as_is" and row.get("persona"):
            return True
    return False


def assign_personas_from_pool(
    episode_ids: list[str],
    pool: list[str],
    *,
    variant: str = "baseline:ppol",
    arm: str = "as_is",
    seed: int = 0,
) -> list[dict[str, Any]]:
    """Round-robin assign pool personas onto episode ids (seeded shuffle of the pool)."""

    if not pool or not episode_ids:
        return []
    order = list(pool)
    random.Random(seed).shuffle(order)
    rows: list[dict[str, Any]] = []
    for i, eid in enumerate(episode_ids):
        rows.append({
            "variant": variant,
            "arm": arm,
            "episode_id": str(eid),
            "persona": order[i % len(order)],
        })
    return rows
