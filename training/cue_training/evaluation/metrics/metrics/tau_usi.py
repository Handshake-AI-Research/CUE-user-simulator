from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path
from typing import Mapping

DEFAULT_TAU_USI_DATASET = "cmu-lti/tau-usi"
DEFAULT_TAU_USI_JSON = "data/tau_bench_tasks_unified.json"


def _resolve_json_path(source: str | None) -> Path:
    """Local path, or download ``data/tau_bench_tasks_unified.json`` from HF ``cmu-lti/tau-usi``.

    ``source`` may be a filesystem path, ``None`` / ``\"hf\"`` / ``\"hf:...\"`` to force the
    HuggingFace dataset download (cached under the HF hub cache).
    """

    if source and not source.startswith("hf:") and source != "hf":
        return Path(source)
    import huggingface_hub

    return Path(
        huggingface_hub.hf_hub_download(
            repo_id=DEFAULT_TAU_USI_DATASET,
            filename=DEFAULT_TAU_USI_JSON,
            repo_type="dataset",
        )
    )


def _rewards_from_payload(payload: object) -> dict[str, float]:
    if not isinstance(payload, Mapping):
        return {}
    rewards: dict[str, float] = {}
    for instance_id, record in payload.items():
        if isinstance(record, Mapping) and isinstance(record.get("reward"), (int, float)):
            rewards[str(instance_id)] = float(record["reward"])
    return rewards


@lru_cache(maxsize=4)
def human_rewards(source: str | None = None) -> dict[str, float]:
    """Map tau-usi instance_id -> human tau2 reward (cached).

    Faithful to MirrorBench's ``metric:env/tau2_task_success``: pull
    ``data/tau_bench_tasks_unified.json`` from ``cmu-lti/tau-usi`` (or a local path)
    and read each record's ``reward``. Returns ``{}`` on any failure so the metric
    degrades to unpaired rather than erroring the run.
    """

    try:
        path = _resolve_json_path(source)
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:  # noqa: BLE001
        return {}
    return _rewards_from_payload(payload)


def human_rewards_or_error(source: str | None = None) -> tuple[dict[str, float], str | None]:
    """Like ``human_rewards``, but returns ``(rewards, error_message)`` instead of swallowing."""

    try:
        path = _resolve_json_path(source)
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:  # noqa: BLE001
        return {}, str(exc)
    rewards = _rewards_from_payload(payload)
    if not rewards:
        return {}, f"no reward fields found in {path}"
    return rewards, None
