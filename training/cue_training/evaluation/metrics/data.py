from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

Turn = dict[str, Any]

_HUMAN_KEYS = (
    "real_conversation",
    "human_conversation",
    "reference_conversation",
    "reference",
    "messages",
    "conversation",
    "conversations",
    "turns",
)
_ROLLOUT_KEYS = (
    "rollout_conversation",
    "proxy_conversation",
    "synthetic_conversation",
    "generated_conversation",
    "rollout",
    "rollout_turns",
)
_USER_ROLES = {"user", "human", "customer"}
_ASSISTANT_ROLES = {"assistant", "agent", "bot", "chatbot", "model"}


@dataclass
class BaselineEpisode:
    episode_id: str
    human: list[Turn]
    base: list[Turn]
    metadata: dict[str, Any]
    raw: dict[str, Any]


@dataclass
class Episode:
    episode_id: str
    human: list[Turn]
    base: list[Turn]
    proxy: list[Turn]
    metadata: dict[str, Any]
    base_metadata: dict[str, Any]
    arm: str = "all"
    domain: str = "all"


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def _norm_role(role: str) -> str | None:
    role = role.lower()
    if role in _USER_ROLES:
        return "user"
    if role in _ASSISTANT_ROLES:
        return "assistant"
    return None


def normalize_turns(value: Any) -> list[Turn]:
    turns: list[Turn] = []
    if not isinstance(value, list):
        return turns
    for item in value:
        if not isinstance(item, Mapping):
            continue
        role = _norm_role(str(item.get("role") or item.get("speaker") or ""))
        if role is None:
            continue
        content = str(item.get("content") or item.get("text") or item.get("message") or "").strip()
        if not content:
            continue
        turn: Turn = {"role": role, "content": content}
        if item.get("metadata"):
            turn["metadata"] = item["metadata"]
        if item.get("raw_data"):
            turn["raw_data"] = item["raw_data"]
        turns.append(turn)
    return turns


def _first_turns(record: Mapping[str, Any], keys: tuple[str, ...]) -> list[Turn]:
    for key in keys:
        turns = normalize_turns(record.get(key))
        if turns:
            return turns
    return []


def episode_key(record: Mapping[str, Any]) -> str:
    eid = str(record.get("episode_id") or record.get("instance_id") or record.get("task_id") or record.get("id") or "")
    return eid.split("::", 1)[0]


def episode_arm(record: Mapping[str, Any]) -> str:
    """Rollout arm (e.g. paired / sample_shuffled / dataset_mean); 'all' when untagged."""

    meta = record.get("metadata") if isinstance(record.get("metadata"), Mapping) else {}
    for source in (meta, record):
        for key in ("arm", "rollout_kind"):
            value = source.get(key)
            if value:
                return str(value)
    return "all"


def episode_domain(record: Mapping[str, Any]) -> str:
    """Task domain (tau2: airline/retail; simarena: writing/math); 'all' when untagged."""

    meta = record.get("metadata") if isinstance(record.get("metadata"), Mapping) else {}
    for source in (meta, record):
        for key in ("domain", "tau2_domain", "dataset"):
            value = source.get(key)
            if value:
                return str(value)
    return "all"


def _metadata(record: Mapping[str, Any]) -> dict[str, Any]:
    meta = dict(record.get("metadata") or {})
    for key in (
        "dataset",
        "domain",
        "tau2_domain",
        "intent",
        "task_description",
        "task_id",
        "instance_id",
        "source_id",
        "agent_id",
        "user_id",
        "rollout_kind",
        "sample_id",
        "arm",
        "persona_origin_episode_id",
        "shuffled_episode_id",
    ):
        if record.get(key) is not None:
            meta.setdefault(key, record[key])
    if "domain" not in meta:
        domain = episode_domain(record)
        if domain != "all":
            meta["domain"] = domain
    # Infer persona origin when older rollouts omit it.
    if "persona_origin_episode_id" not in meta:
        arm = str(meta.get("arm") or episode_arm(record))
        if arm == "paired":
            eid = episode_key(record)
            if eid:
                meta["persona_origin_episode_id"] = eid
        elif arm == "sample_shuffled":
            if meta.get("shuffled_episode_id"):
                meta["persona_origin_episode_id"] = str(meta["shuffled_episode_id"])
    return meta


def load_baseline(path: Path) -> dict[str, BaselineEpisode]:
    out: dict[str, BaselineEpisode] = {}
    for record in read_jsonl(path):
        key = episode_key(record)
        if not key:
            continue
        human = _first_turns(record, _HUMAN_KEYS)
        base = _first_turns(record, _ROLLOUT_KEYS)
        if not human or not base:
            continue
        out[key] = BaselineEpisode(key, human=human, base=base, metadata=_metadata(record), raw=record)
    return out


def load_baseline_paths(
    paths: list[Path],
    *,
    sim_labels: list[str] | None = None,
    seed: int = 0,
) -> dict[str, BaselineEpisode]:
    """Load floor-sim baselines from multiple simulator JSONLs for aggregate fitting.

    Humans are taken once per real ``episode_id`` (first file wins). For each episode,
    exactly one floor-sim backbone is chosen with a seeded, episode-stable draw among the
    sims that have that episode -- so classes stay 1:1 and humans do not reappear once per
    backbone. The chosen sim is recorded in ``metadata["aggregate_sim"]``. Scoring still
    joins on the real episode id via :func:`load_baseline` / :func:`load_episodes`.
    """

    if not paths:
        raise ValueError("load_baseline_paths requires at least one baseline path")
    labels = list(sim_labels) if sim_labels is not None else [p.parent.name for p in paths]
    if len(labels) != len(paths):
        raise ValueError(f"sim_labels length {len(labels)} != paths length {len(paths)}")

    humans: dict[str, list[Turn]] = {}
    human_meta: dict[str, dict[str, Any]] = {}
    human_raw: dict[str, dict[str, Any]] = {}
    # episode_id -> sim -> base turns
    bases: dict[str, dict[str, list[Turn]]] = {}
    for path, sim in zip(paths, labels, strict=True):
        for record in read_jsonl(path):
            key = episode_key(record)
            if not key:
                continue
            human = _first_turns(record, _HUMAN_KEYS)
            base = _first_turns(record, _ROLLOUT_KEYS)
            if not human or not base:
                continue
            if key not in humans:
                humans[key] = human
                human_meta[key] = _metadata(record)
                human_raw[key] = dict(record)
            bases.setdefault(key, {})[sim] = base

    out: dict[str, BaselineEpisode] = {}
    for key in sorted(humans):
        available = sorted(bases.get(key, {}))
        if not available:
            continue
        # Episode-stable pick: same (seed, episode_id, available set) always chooses the same sim.
        digest = hashlib.md5(f"{seed}:{key}:{','.join(available)}".encode("utf-8")).hexdigest()
        sim = available[int(digest, 16) % len(available)]
        meta = dict(human_meta[key])
        meta["aggregate_sim"] = sim
        meta["aggregate_source_episode_id"] = key
        fit_key = f"{key}__{sim}"
        out[fit_key] = BaselineEpisode(
            fit_key,
            human=humans[key],
            base=bases[key][sim],
            metadata=meta,
            raw=human_raw[key],
        )
    return out


def load_episodes(baseline_path: Path, candidate_path: Path | None) -> tuple[list[Episode], dict[str, BaselineEpisode]]:
    baseline = load_baseline(baseline_path)
    # Score every row of the file being evaluated. A single rollout file may hold multiple
    # arms (paired / sample_shuffled / ...) as separate rows sharing one episode_id, so read
    # the raw rows (not the arm-deduped baseline dict) to keep each arm as its own episode.
    rows = read_jsonl(candidate_path or baseline_path)
    episodes: list[Episode] = []
    for record in rows:
        key = episode_key(record)
        base_ep = baseline.get(key)
        if base_ep is None:
            continue
        proxy = _first_turns(record, _ROLLOUT_KEYS)
        if not proxy:
            continue
        meta = _metadata(record)
        episodes.append(
            Episode(
                episode_id=key,
                human=base_ep.human,
                base=base_ep.base,
                proxy=proxy,
                metadata=meta,
                base_metadata=base_ep.metadata,
                arm=episode_arm(record),
                domain=episode_domain(record),
            )
        )
    return episodes, baseline


def user_text(turns: list[Turn]) -> str:
    return "\n".join(str(t.get("content") or "") for t in turns if t.get("role") == "user").strip()


def user_turn_texts(turns: list[Turn]) -> list[str]:
    """Individual (non-empty) user-turn strings, for per-turn style aggregation."""

    return [c for t in turns if t.get("role") == "user" and (c := str(t.get("content") or "").strip())]


def user_turn_documents(turns: list[Turn], *, neighbor_radius: int = 0) -> list[str]:
    """LUAR documents from user turns, optionally with adjacent user-turn context.

    ``neighbor_radius=1`` makes each document the centered user turn plus the previous
    and next *user* turns (assistant turns are excluded to avoid task leakage).
    """

    texts = user_turn_texts(turns)
    radius = max(0, int(neighbor_radius))
    if radius == 0:
        return texts
    docs: list[str] = []
    for i in range(len(texts)):
        lo = max(0, i - radius)
        hi = min(len(texts), i + radius + 1)
        docs.append("\n".join(texts[lo:hi]))
    return docs


# Default for LUAR authorship metrics: one document per user turn (no overlap).
# LUAR jointly self-attends over the excerpt set, so windows are unnecessary.
DEFAULT_LUAR_NEIGHBOR_RADIUS = 0


def assistant_text(turns: list[Turn]) -> str:
    return "\n".join(str(t.get("content") or "") for t in turns if t.get("role") == "assistant").strip()

