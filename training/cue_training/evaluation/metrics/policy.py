"""Method/arm skip rules for the default metrics suite."""

from __future__ import annotations

# Sampled / unconditioned arms (no grounded same-episode human for mimicry retrieval).
SAMPLED_ARMS = frozenset({
    "as_is",
    "sample_diverse",
    "sample_random",
    "sampled_uncond",
    "sampled_pop",
    "dataset_mean",
})

CUE_METHODS = frozenset({"general", "refined", "proposals", "cue", "cue-general", "cue-refined"})


def _norm_method(method: str | None) -> str:
    return str(method or "").strip().lower().replace("-", "_")


def _is_cue_method(kind: str | None, method: str | None) -> bool:
    if str(kind or "").strip().lower() == "cue":
        return True
    return _norm_method(method) in CUE_METHODS or _norm_method(method).startswith("cue_")


def skip_ava_mrr(*, kind: str | None, method: str | None, arm: str | None) -> str | None:
    """Return skip reason for Wegmann AVA / full-corpus MRR, else None."""

    m = _norm_method(method)
    a = str(arm or "").strip()
    if m == "realusersim":
        return "paper realusersim is sampled WildChat (use realusersim_paired_noex for paired)"
    if m == "userlm":
        return "userlm has no paired human grounding for authorship mimicry"
    if m == "usp" and a == "sample_diverse":
        return "usp sample_diverse is not episode-paired"
    if _is_cue_method(kind, method) and a in SAMPLED_ARMS - {"as_is"}:
        return f"cue sampled arm {a!r} has no grounded Th"
    if _is_cue_method(kind, method) and a == "as_is":
        return "cue as_is skipped for AVA/MRR"
    return None


def skip_paired_audit(*, kind: str | None, method: str | None, arm: str | None) -> str | None:
    """Skip when there is no grounded same-episode human trajectory."""

    m = _norm_method(method)
    a = str(arm or "").strip()
    if m == "usp" and a == "sample_diverse":
        return "usp sample_diverse is not episode-paired"
    if _is_cue_method(kind, method) and a in SAMPLED_ARMS:
        return f"cue arm {a!r} has no grounded Th"
    return None