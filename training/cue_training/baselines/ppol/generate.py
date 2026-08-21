"""Run the evolved PPol generator ``G(c, D, N)`` from a frozen ``best_program.py``.

Faithful PPol: personas are produced by the OpenEvolve-evolved Python program
(``persona_policies/evolution/initial_generator.py`` after evolution), not by our
old genome approximation. This module loads that program from a file path, routes
its LLM calls through our provider (litellm), and returns the per-persona
``expanded_instruction`` strings that steer the user simulator.

The generator's LLM model is resolved by the upstream plumbing
(``_generator_utils._llm_model_id`` -> ``PersonaPoliciesConfig().llm_model``); we
override it at runtime by setting the module-level cache so we never edit the
vendored submodule. ``api_base``/key are passed via env (litellm reads
``HOSTED_VLLM_API_BASE``/``OPENAI_API_KEY``), matching how ``completion_text`` calls
``litellm.completion``.
"""

from __future__ import annotations

import importlib.util
import os
import sys
from functools import lru_cache
from pathlib import Path
from types import ModuleType
from typing import Any

# Repo root of the vendored persona-policies submodule (parent of persona_policies/).
_VENDOR_ROOT = Path(__file__).resolve().parent / "external" / "persona-policies"


def _ensure_vendor_on_path() -> None:
    root = str(_VENDOR_ROOT)
    if root not in sys.path:
        sys.path.insert(0, root)


def _route_generation_llm(model: str, api_base: str | None, api_key_env: str | None) -> None:
    """Point the upstream generator's LLM at our provider without editing the submodule.

    ``_generator_utils._chat`` reads a cached model id; overriding that global makes
    both the population and roleplay-expansion calls use ``model``. litellm resolves
    credentials from the environment, so for a vLLM-served model we set
    ``HOSTED_VLLM_API_BASE`` (litellm's ``hosted_vllm`` provider reads it when no
    explicit ``api_base`` is passed).
    """

    _ensure_vendor_on_path()
    from persona_policies.evolution import _generator_utils

    _generator_utils._cached_llm_model = model  # noqa: SLF001
    if api_base and model.startswith("hosted_vllm/"):
        os.environ.setdefault("HOSTED_VLLM_API_BASE", api_base)
        if api_key_env:
            os.environ.setdefault("HOSTED_VLLM_API_KEY", os.getenv(api_key_env, "EMPTY"))


@lru_cache(maxsize=8)
def _load_program(best_program_path: str) -> ModuleType:
    """Import an evolved ``best_program.py`` as a standalone module (cached per path)."""

    _ensure_vendor_on_path()
    path = Path(best_program_path)
    if not path.is_file():
        raise FileNotFoundError(
            f"PPol best_program.py not found at {path}. Run the evolution pipeline first "
            "(scripts/train_ppol.sh) and point PPOL_BEST_PROGRAM at the evolved program."
        )
    spec = importlib.util.spec_from_file_location(f"ppol_best_{abs(hash(str(path)))}", path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load PPol program from {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    if not hasattr(module, "generate_personas_detailed"):
        raise AttributeError(
            f"{path} has no generate_personas_detailed(c, D, N); not a valid PPol generator."
        )
    return module


def generate_personas(
    scenario: str,
    n: int,
    *,
    best_program_path: str,
    model: str = "openrouter/google/gemini-3-flash-preview",
    api_base: str | None = None,
    api_key_env: str | None = "OPENAI_API_KEY",
) -> list[str]:
    """Return ``n`` roleplay-instruction strings for one task scenario ``c``.

    Each string is a persona's ``expanded_instruction`` (the block appended to the
    user-simulator system prompt). Personas without an expansion are dropped.
    """

    _route_generation_llm(model, api_base, api_key_env)
    module = _load_program(best_program_path)
    axes = getattr(module, "DIVERSITY_AXES", None) or []
    personas: list[dict[str, Any]] = module.generate_personas_detailed(scenario, axes, int(n))
    out: list[str] = []
    for p in personas:
        instruction = str((p or {}).get("expanded_instruction") or "").strip()
        if instruction:
            out.append(instruction)
    return out
