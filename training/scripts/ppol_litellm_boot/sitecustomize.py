"""Auto-loaded litellm bootstrap for the PPol venv (put this dir first on PYTHONPATH).

vLLM-served models aren't in litellm's price/context map, so litellm raises
"This model isn't mapped yet ... custom_llm_provider=hosted_vllm" when tau2 does
cost / model-info lookups. We register the served model ids (from ``PPOL_VLLM_MODELS``)
with a generous context window and zero cost so those lookups succeed. Fully
defensive: any failure here must never break interpreter startup.
"""

from __future__ import annotations

import os


def _patch_persona_config() -> None:
    """Honor model-routing env overrides even if the upstream config.py lacks the hooks.

    The cloned PersonaPoliciesConfig defaults several models to OpenRouter
    (llm_model, evolution_feedback_model, tau2_llm_nl_assertions, tau2_llm_env_interface).
    Editing config.py doesn't propagate across machines (it's external checkout content), so
    we wrap __post_init__ here (main repo, auto-loaded via PYTHONPATH) to apply env overrides
    to every config instance. Defensive: any failure is ignored.
    """

    overrides = {
        "PERSONA_POLICIES_LLM_MODEL": "llm_model",
        "PERSONA_POLICIES_FEEDBACK_MODEL": "evolution_feedback_model",
        "PERSONA_POLICIES_NL_ASSERTIONS": "tau2_llm_nl_assertions",
        "PERSONA_POLICIES_ENV_INTERFACE": "tau2_llm_env_interface",
    }
    if not any(os.environ.get(k) for k in overrides):
        return
    try:
        from persona_policies.config import PersonaPoliciesConfig
    except Exception:  # noqa: BLE001
        return
    _orig = PersonaPoliciesConfig.__post_init__

    def _post_init(self) -> None:  # type: ignore[no-untyped-def]
        _orig(self)
        for env_key, attr in overrides.items():
            val = os.environ.get(env_key, "").strip()
            if val:
                setattr(self, attr, val)

    try:
        PersonaPoliciesConfig.__post_init__ = _post_init  # type: ignore[assignment]
    except Exception:  # noqa: BLE001
        pass


def _register() -> None:
    ids = os.environ.get("PPOL_VLLM_MODELS", "").strip()
    if not ids:
        return
    try:
        import litellm
    except Exception:  # noqa: BLE001
        return
    max_tokens = int(os.environ.get("PPOL_VLLM_MAX_TOKENS", "131072"))
    entries: dict[str, dict] = {}
    for raw in ids.split(","):
        bare = raw.strip()
        if not bare:
            continue
        bare = bare[len("hosted_vllm/"):] if bare.startswith("hosted_vllm/") else bare
        info = {
            "max_tokens": max_tokens,
            "max_input_tokens": max_tokens,
            "max_output_tokens": max_tokens,
            "input_cost_per_token": 0.0,
            "output_cost_per_token": 0.0,
            "litellm_provider": "hosted_vllm",
            "mode": "chat",
        }
        # Register under both the bare id and the hosted_vllm/-prefixed id so cost/model-info
        # lookups succeed regardless of which form the caller passes.
        entries[bare] = info
        entries[f"hosted_vllm/{bare}"] = dict(info)
    if not entries:
        return
    try:
        litellm.register_model(entries)
    except Exception:  # noqa: BLE001
        pass


_patch_persona_config()
_register()
