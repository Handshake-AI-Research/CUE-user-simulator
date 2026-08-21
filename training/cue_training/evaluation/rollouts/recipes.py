"""Baseline/CUE method manifests: arms, artifacts, and serving requirements."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

RolloutType = Literal["baseline", "cue"]
ServeKind = Literal["none", "sim_vllm", "policy_vllm", "sidecar", "sim_vllm+sidecar"]


BASELINE_METHODS = (
    "base",
    "userlm",
    "realusersim",
    "realusersim_paired_noex",
    "usp",
    "ppol",
)

CUE_METHODS = ("proposals", "general", "refined")


@dataclass(frozen=True)
class MethodRecipe:
    name: str
    rollout_type: RolloutType
    variant: str
    default_arms: tuple[str, ...]
    needs_cue_embeddings: bool = False
    needs_sidecar: bool = False
    needs_policy_vllm: bool = False
    needs_sim_vllm: bool = False
    train_supported: bool = False
    paper_preset: str | None = None
    required_artifacts: tuple[str, ...] = ()
    notes: str = ""


RECIPES: dict[tuple[str, str], MethodRecipe] = {
    ("baseline", "base"): MethodRecipe(
        name="base",
        rollout_type="baseline",
        variant="base",
        default_arms=("as_is",),
        needs_sim_vllm=True,
        notes="API or local floor model; no training.",
    ),
    ("baseline", "userlm"): MethodRecipe(
        name="userlm",
        rollout_type="baseline",
        variant="baseline:userlm",
        default_arms=("as_is",),
        needs_policy_vllm=True,
        needs_sidecar=True,
        train_supported=True,
        paper_preset="userlm_released",
        required_artifacts=("model_id.txt",),
        notes="Inference-only microsoft/UserLM-8b with paper decoding.",
    ),
    ("baseline", "realusersim"): MethodRecipe(
        name="realusersim",
        rollout_type="baseline",
        variant="baseline:realusersim",
        default_arms=("as_is",),
        needs_sim_vllm=True,
        train_supported=True,
        paper_preset="realusersim_gpt4o_full",
        required_artifacts=("profiles.jsonl", "manifest.json"),
        notes=(
            "Paper τ-bench setup: randomly sample a WildChat-grounded profile per task "
            "(as_is). Personas come from train_paper WildChat curation, not eval-set extraction."
        ),
    ),
    ("baseline", "realusersim_paired_noex"): MethodRecipe(
        name="realusersim_paired_noex",
        rollout_type="baseline",
        variant="baseline:realusersim_paired_noex",
        default_arms=("paired",),
        needs_sim_vllm=True,
        train_supported=False,
        notes=(
            "Paired RealUserSim: GPT-4o extract from each eval real_conversation, with "
            "verbatim command examples stripped (paper leakage control). Loads the "
            "RealUserSim simulator via registry alias."
        ),
    ),
    ("baseline", "usp"): MethodRecipe(
        name="usp",
        rollout_type="baseline",
        variant="baseline:usp",
        default_arms=("paired", "sample_diverse"),
        needs_policy_vllm=True,
        needs_sidecar=True,
        train_supported=True,
        paper_preset="usp_paper_full",
        required_artifacts=("profiles.jsonl", "manifest.json"),
    ),
    ("baseline", "ppol"): MethodRecipe(
        name="ppol",
        rollout_type="baseline",
        variant="baseline:ppol",
        default_arms=("as_is",),
        needs_sim_vllm=True,
        train_supported=True,
        paper_preset="ppol_openevolve",
        required_artifacts=("best_program.py", "manifest.json"),
    ),
    ("cue", "proposals"): MethodRecipe(
        name="proposals",
        rollout_type="cue",
        variant="baseline:realusersim",
        default_arms=("paired", "sample_shuffled"),
        needs_sim_vllm=True,
        notes=(
            "data annotation extraction on real eval conversations (tau2/simarena/prism) + RealUserSim "
            "sidecars. Arms are always paired/sample_shuffled (ignores cue.arms sampled_*)."
        ),
    ),
    ("cue", "general"): MethodRecipe(
        name="general",
        rollout_type="cue",
        variant="decoder",
        default_arms=("paired", "sample_shuffled"),
        needs_sidecar=True,
        needs_sim_vllm=True,
        notes=(
            "joint training live trajectory encode + unified decode + simulator steering. "
            "Arms sampled_uncond/sampled_pop auto-export the paired population when needed."
        ),
    ),
    ("cue", "refined"): MethodRecipe(
        name="refined",
        rollout_type="cue",
        variant="decoder",
        default_arms=("paired", "sample_shuffled"),
        needs_cue_embeddings=True,
        needs_sidecar=True,
        needs_sim_vllm=True,
        notes=(
            "Simulator-specific decoder refinement checkpoint; hard-fail if missing. "
            "Supports sampled_uncond/sampled_pop via cue.sampler_checkpoint."
        ),
    ),
}

def expand_methods(rollout_type: str, methods: list[str]) -> list[str]:
    key = rollout_type.strip().lower()
    if key not in {"baseline", "cue"}:
        raise ValueError(f"rollout type must be baseline|cue, got {rollout_type!r}")
    catalog = BASELINE_METHODS if key == "baseline" else CUE_METHODS
    aliases = {
        "base_api": "base",
        "base_local": "base",
        "real_user_sim": "realusersim",
        "realusersim-paired-noex": "realusersim_paired_noex",
    }
    if not methods or methods == ["all"]:
        return list(catalog)
    out: list[str] = []
    for raw in methods:
        name = aliases.get(raw.strip().lower().replace("-", "_"), raw.strip().lower().replace("-", "_"))
        if name == "all":
            for item in catalog:
                if item not in out:
                    out.append(item)
            continue
        if name not in catalog:
            raise ValueError(
                f"unknown {key} method {raw!r}. Choose from: {', '.join(catalog)} (or all)"
            )
        if name not in out:
            out.append(name)
    return out


def get_recipe(rollout_type: str, method: str) -> MethodRecipe:
    try:
        return RECIPES[(rollout_type, method)]
    except KeyError as exc:
        raise ValueError(f"no recipe for {rollout_type}/{method}") from exc


def harness_variant(recipe: MethodRecipe, *, simulator_kind: str) -> str:
    """Map a recipe to the underlying tau2/simarena variant string."""

    if recipe.name == "base":
        return "base_local" if simulator_kind == "vllm" else "base_api"
    return recipe.variant
