"""Labels that count as user-simulator / environment faults (vs agent faults)."""

from __future__ import annotations

# Split of the former umbrella "Critical User Simulator Error".
SIMULATOR_ERROR_LABELS: tuple[str, ...] = (
    "Premature User Stop",
    "User Identity / Task Derailment",
    "Other Simulator Error",
)

# Kept so old tagged rows / CLI merges still resolve during transition.
LEGACY_SIMULATOR_ERROR_LABEL = "Critical User Simulator Error"

ALL_SIMULATOR_ERROR_LABELS: tuple[str, ...] = (
    *SIMULATOR_ERROR_LABELS,
    LEGACY_SIMULATOR_ERROR_LABEL,
)

ENVIRONMENT_ERROR_LABEL = "Environment Error"

# Default paper cut shared by ``cue-fma plot`` and ``cue-fma tvd`` (caller overrides win).
# Early Stop → Env; Identity / Leakage / legacy Critical → Other Sim (= plot User Error).
PAPER_CUT_MERGE: dict[str, str] = {
    "Premature User Stop": ENVIRONMENT_ERROR_LABEL,
    "User Identity / Task Derailment": "Other Simulator Error",
    "User Data Leakage": "Other Simulator Error",
    LEGACY_SIMULATOR_ERROR_LABEL: "Other Simulator Error",
}

# Paper "User Error" bucket (Early Stop / Other Sim / Leakage / Identity Derail + legacy).
USER_ERROR_LABELS: tuple[str, ...] = (
    *ALL_SIMULATOR_ERROR_LABELS,
    "User Data Leakage",
)

# Folded into ``Critical User Simulator Error`` via ``cue-fma plot|tvd --user-error``.
USER_ERROR_BUCKET_SOURCES: tuple[str, ...] = (
    "Premature User Stop",
    "Other Simulator Error",
    "User Data Leakage",
    "User Identity / Task Derailment",
)
USER_ERROR_BUCKET_TARGET = LEGACY_SIMULATOR_ERROR_LABEL
