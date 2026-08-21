"""Idempotent registration of our custom user into tau2's global registry."""

from __future__ import annotations

from cue_training.evaluation.tau2_bench.users import USER_NAME, build_cue_eval_user_class


def register_cue_eval_user() -> str:
    """Register (once) our ``cue_eval_user`` in tau2's registry; return its name.

    ``registry.register_user`` raises on duplicate names, so this no-ops if the name
    is already present and only errors on a genuine name/class conflict.
    """

    from tau2.registry import registry  # type: ignore

    user_cls = build_cue_eval_user_class()
    if USER_NAME in registry.get_users():
        return USER_NAME
    registry.register_user(user_cls, USER_NAME)
    return USER_NAME
