"""Ingest failed tau2 human + simulator trajectories into ``corpus.jsonl``."""

from __future__ import annotations

import json
import re
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any

from cue_training.evaluation.common.io import read_jsonl, write_jsonl
from cue_training.evaluation.fma.conversation import normalize_conversation
from cue_training.evaluation.fma.criteria import default_tasks_dir, load_task_criteria
from cue_training.evaluation.fma.paths import corpus_path, ensure_run
from cue_training.evaluation.fma.store import write_json
from cue_training.evaluation.tau2_bench.data import (
    parse_tau2_domain,
    resolve_tau2_task_ids,
)

SUCCESS_THRESHOLD = 1.0 - 1e-6


def _meta(record: Mapping[str, Any]) -> dict[str, Any]:
    meta = record.get("metadata") or {}
    return dict(meta) if isinstance(meta, Mapping) else {}


def _clean(val: Any) -> str:
    if val is None:
        return ""
    s = str(val).strip()
    return "" if s.lower() == "nan" else s


# tau-usi wraps the tau2 user instruction in a fixed role-play template whose trailer carries
# survey/runtime boilerplate and a misleading "**Task index:** N" (N is the tau-usi instance
# index, not the tau2 task id). Keep only the instruction core so prompts see the real task.
_TASK_CORE_RE = re.compile(
    r"agent\.\s*(.*?)\s*\*\*You may start with", re.DOTALL | re.IGNORECASE
)


def _task_core(text: str) -> str:
    match = _TASK_CORE_RE.search(text or "")
    return match.group(1).strip() if match else text


def _task_description(record: Mapping[str, Any]) -> str:
    return _task_core(
        _clean(record.get("task_description") or record.get("intent") or "")
    )


def _domain(record: Mapping[str, Any], meta: Mapping[str, Any]) -> str:
    return (
        _clean(meta.get("tau2_domain"))
        or _clean(meta.get("domain"))
        or _clean(record.get("dataset"))
        or "unknown"
    )


def _episode_id(record: Mapping[str, Any]) -> str:
    return _clean(record.get("episode_id") or record.get("task_id") or "")


def _task_id(record: Mapping[str, Any], meta: Mapping[str, Any]) -> str:
    return _clean(
        meta.get("tau2_task_id") or record.get("task_id") or _episode_id(record)
    )


def load_human_rewards(source: str | None) -> dict[str, float]:
    """Map episode/instance id -> reward from local JSON or HF tau-usi."""

    if source and not str(source).startswith("hf:"):
        path = Path(source).expanduser()
        payload = json.loads(path.read_text(encoding="utf-8"))
    else:
        import huggingface_hub

        path = Path(
            huggingface_hub.hf_hub_download(
                repo_id="cmu-lti/tau-usi",
                filename="data/tau_bench_tasks_unified.json",
                repo_type="dataset",
            )
        )
        payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping):
        return {}
    rewards: dict[str, float] = {}
    for instance_id, record in payload.items():
        # Flat map {id: 0.0} or tau-usi unified {id: {reward: 0.0, ...}}.
        if isinstance(record, (int, float)):
            rewards[str(instance_id)] = float(record)
        elif isinstance(record, Mapping) and isinstance(
            record.get("reward"), (int, float)
        ):
            rewards[str(instance_id)] = float(record["reward"])
    return rewards


def _human_reward(rewards: dict[str, float], episode_id: str) -> float | None:
    if episode_id in rewards:
        return rewards[episode_id]
    if "_ann" in episode_id:
        base = episode_id.rsplit("_ann", 1)[0]
        if base in rewards:
            return rewards[base]
    return None


def _conversation_from_record(
    record: Mapping[str, Any], *, human: bool
) -> list[dict[str, Any]]:
    meta = _meta(record)
    if not human:
        full = meta.get("full_conversation")
        if isinstance(full, list) and full:
            return normalize_conversation(full)
        return normalize_conversation(record.get("rollout_conversation") or [])
    return normalize_conversation(record.get("real_conversation") or [])


def _sim_failed(meta: Mapping[str, Any]) -> bool | None:
    if "is_success" in meta and meta["is_success"] is not None:
        return not bool(meta["is_success"])
    reward = meta.get("reward")
    if isinstance(reward, (int, float)):
        return float(reward) < SUCCESS_THRESHOLD
    return None


def ingest_human(
    path: Path,
    *,
    rewards: dict[str, float],
    criteria: dict[tuple[str, str], str],
    source_id: str = "human",
    tasks_dir: str | None = None,
) -> list[dict[str, Any]]:
    records = list(read_jsonl(str(path)))
    # tau-usi instance ids are NOT tau2 task ids (airline_0 is tau2 airline task 20), so join
    # each episode to its tau2 task by task_description content, same as the rollout harness.
    episodes = [
        {
            "episode_id": _episode_id(r),
            "tau2_domain": parse_tau2_domain(_episode_id(r), _meta(r)),
            "task_description": _clean(
                r.get("task_description") or r.get("intent") or ""
            ),
        }
        for r in records
    ]
    domains_dir = Path(tasks_dir).expanduser() if tasks_dir else default_tasks_dir()
    resolve_tau2_task_ids(episodes, domains_dir.parents[2])
    resolved = {
        e["episode_id"]: (e["tau2_domain"], e["tau2_task_id"]) for e in episodes
    }
    rows: list[dict[str, Any]] = []
    for record in records:
        eid = _episode_id(record)
        if not eid:
            continue
        reward = _human_reward(rewards, eid)
        if reward is None or reward >= SUCCESS_THRESHOLD:
            continue
        meta = _meta(record)
        match_domain, match_tid = resolved.get(eid, (None, None))
        domain = match_domain or _domain(record, meta)
        tau2_tid = match_tid or ""
        conv = _conversation_from_record(record, human=True)
        if not conv:
            continue
        rows.append(
            {
                "primary_key": f"{source_id}:{eid}",
                "episode_id": eid,
                "source": "human",
                "source_id": source_id,
                "arm": None,
                "variant": None,
                "domain": domain,
                "task_id": tau2_tid,
                "task_description": _task_description(record),
                "success_criteria": criteria.get((domain, tau2_tid), ""),
                "reward": reward,
                "is_success": False,
                "reward_report": None,
                "conversation": conv,
                "failed": True,
            }
        )
    return rows


def _seed_from_path(path: Path) -> int | None:
    """Return ``N`` from a ``.../seed-N/...`` rollout path, else None."""

    for part in path.parts:
        if part.startswith("seed-") and part[5:].isdigit():
            return int(part[5:])
    return None


def _source_id_from_path(path: Path) -> str:
    """``<method>-<simulator>`` from a rollout job path.

    Prefers the MirrorBench layout
    ``.../{baseline|cue}/{method}/{simulator}/rollout.*.jsonl`` so multiseed trees
    like ``rollouts/seed-0/baseline-seed-0/.../ppol/llama/`` still yield ``ppol-llama``
    (not ``seed-0-llama``). Falls back to ``<experiment>-<simulator>`` for flat
    ``rollouts/<experiment>/.../<simulator>/`` trees, then to the file stem.
    Seed wrappers are intentionally omitted from the id so TVD can pool raw counts
    across seeds under one ``source_id``.
    """

    parts = path.parts
    # .../{baseline|cue}/{method}/{simulator}/rollout.*.jsonl
    if len(parts) >= 4 and parts[-4] in ("baseline", "cue"):
        kind, method, simulator = parts[-4], parts[-3], parts[-2]
        if kind == "cue":
            return f"cue-{method}-{simulator}"
        return f"{method}-{simulator}"

    if "rollouts" in parts:
        i = parts.index("rollouts")
        experiment = parts[i + 1] if len(parts) > i + 1 else ""
        # Skip seed-N wrappers used by multiseed metrics layout.
        if (
            experiment.startswith("seed-")
            and experiment[5:].isdigit()
            and len(parts) > i + 2
        ):
            experiment = parts[i + 2]
        simulator = path.parent.name
        if experiment and simulator and simulator != experiment:
            return f"{experiment}-{simulator}"
        if experiment:
            return experiment
    return path.stem


def ingest_sim(
    path: Path,
    *,
    criteria: dict[tuple[str, str], str],
    source_id: str | None = None,
    arms: set[str] | None = None,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    sid = source_id or _source_id_from_path(path)
    seed = _seed_from_path(path)
    for record in read_jsonl(str(path)):
        meta = _meta(record)
        failed = _sim_failed(meta)
        if failed is not True:
            continue
        arm = _clean(meta.get("arm"))
        if arms is not None and arm and arm not in arms:
            continue
        eid = _episode_id(record)
        if not eid:
            continue
        domain = _domain(record, meta)
        tid = _task_id(record, meta)
        conv = _conversation_from_record(record, human=False)
        if not conv:
            continue
        variant = _clean(meta.get("variant"))
        # Arm stays in the source_id (TVD pools raw counts by this key across seeds).
        # Seed is only in primary_key so the same episode from seed-0/1/2 stays distinct
        # for tagging / review.
        label = f"{sid}|{arm}" if arm else sid
        if seed is not None:
            primary_key = f"{label}:seed-{seed}:{eid}"
        else:
            primary_key = f"{label}:{eid}"
        rows.append(
            {
                "primary_key": primary_key,
                "episode_id": eid,
                "source": "sim",
                "source_id": label,
                "seed": seed,
                "arm": arm or None,
                "variant": variant or None,
                "domain": domain,
                "task_id": tid,
                "task_description": _task_description(record),
                "success_criteria": criteria.get((domain, tid), ""),
                "reward": float(meta["reward"])
                if isinstance(meta.get("reward"), (int, float))
                else None,
                "is_success": False,
                "reward_report": meta.get("reward_report"),
                "conversation": conv,
                "failed": True,
            }
        )
    return rows


def ingest(
    *,
    out_dir: Path,
    human_path: Path | None = None,
    sim_paths: Iterable[Path] = (),
    rewards_source: str | None = None,
    tasks_dir: str | None = None,
    arms: list[str] | None = None,
) -> dict[str, Any]:
    run_dir = ensure_run(out_dir)
    criteria = load_task_criteria(tasks_dir)
    arm_set = set(arms) if arms else None
    rows: list[dict[str, Any]] = []
    if human_path is not None:
        rewards = load_human_rewards(rewards_source)
        rows.extend(
            ingest_human(
                Path(human_path),
                rewards=rewards,
                criteria=criteria,
                tasks_dir=tasks_dir,
            )
        )
    for path in sim_paths:
        rows.extend(ingest_sim(Path(path), criteria=criteria, arms=arm_set))
    write_jsonl(str(corpus_path(run_dir)), rows)
    coverage = {
        "n_failures": len(rows),
        "by_source": {},
        "episode_ids_by_source": {},
    }
    for row in rows:
        sid = row["source_id"]
        coverage["by_source"][sid] = coverage["by_source"].get(sid, 0) + 1
        coverage["episode_ids_by_source"].setdefault(sid, []).append(row["episode_id"])
    write_json(run_dir / "coverage.json", coverage)
    return {
        "run_dir": str(run_dir),
        "n_failures": len(rows),
        "by_source": coverage["by_source"],
    }
