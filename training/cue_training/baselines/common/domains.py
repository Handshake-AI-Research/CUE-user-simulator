"""The three evaluation domains and their vendored normalized data."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

# repo root = CUE/ (this file is CUE/baselines/common/domains.py)
_REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DATA_ROOT = _REPO_ROOT / "data" / "mirrorbench" / "data"


@dataclass(frozen=True)
class Domain:
    key: str  # canonical key used in outputs/metadata
    dataset: str  # value of the normalized "dataset" field
    subdir: str  # folder under the data root


DOMAINS: dict[str, Domain] = {
    "coding": Domain("coding", "openhands_feedback", "openhands_feedback"),
    "customer_service": Domain("customer_service", "tau_usi", "tau_usi"),
    "writing": Domain("writing", "simulator_arena", "simulatorarena"),
    "diversity": Domain("diversity", "prism", "prism"),
}

# Accept common spellings/aliases on the CLI.
_ALIASES: dict[str, str] = {
    "coding": "coding",
    "code": "coding",
    "openhands": "coding",
    "openhands_feedback": "coding",
    "openhands-feedback": "coding",
    "customer_service": "customer_service",
    "customer-service": "customer_service",
    "customerservice": "customer_service",
    "cs": "customer_service",
    "tau": "customer_service",
    "tau_usi": "customer_service",
    "tau-usi": "customer_service",
    "writing": "writing",
    "write": "writing",
    "simulator_arena": "writing",
    "simulator-arena": "writing",
    "simulatorarena": "writing",
    "diversity": "diversity",
    "prism": "diversity",
}


def resolve_domains(name: str) -> list[Domain]:
    """Map a CLI domain name (or ``all``) to ``Domain`` objects."""

    if name in ("core", "three", "main"):
        return [DOMAINS["coding"], DOMAINS["customer_service"], DOMAINS["writing"]]
    if name in ("all", "", None):
        return list(DOMAINS.values())
    key = _ALIASES.get(name.strip().lower())
    if key is None:
        valid = ", ".join(sorted(set(_ALIASES))) + ", all"
        raise ValueError(f"Unknown domain {name!r}. Valid options: {valid}")
    return [DOMAINS[key]]


def normalized_path(domain: Domain, data_root: Path | None = None) -> Path:
    root = Path(data_root) if data_root is not None else DEFAULT_DATA_ROOT
    return root / domain.subdir / "normalized.jsonl"
