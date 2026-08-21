"""Build tau2 eval cases from tau_usi normalized data + CUE embeddings.

Joins each tau_usi episode to its tau2 task id + sub-domain, both parsed from the tau-usi
instance id (``{domain}_{tau2_task_id}[_annN]``), and attaches the per-arm CUE embedding
(paired / sample_shuffled / dataset_mean). Fails closed: episodes without a resolvable
tau2 task id / domain / embedding are reported as skipped, never fabricated.
"""

from __future__ import annotations

import re
from collections import defaultdict
from pathlib import Path
from typing import Any

from cue_training.evaluation.common.io import read_jsonl

from cue_training.evaluation.common.conditioning import CUE_ARMS as _SHARED_CUE_ARMS

TAU2_DOMAINS = ("airline", "retail", "telecom")
CUE_ARMS = _SHARED_CUE_ARMS
# Arms that do not attach a CUE embedding (unconditioned / persona-sidecar baselines).
NO_EMBED_ARMS = frozenset({"as_is"})


def parse_tau2_domain(instance_id: str, metadata: dict[str, Any]) -> str | None:
    explicit = str((metadata or {}).get("tau2_domain") or (metadata or {}).get("domain") or "")
    if explicit in TAU2_DOMAINS:
        return explicit
    prefix = str(instance_id).split("_", 1)[0].lower()
    return prefix if prefix in TAU2_DOMAINS else None


_ANN_SUFFIX = re.compile(r"_ann\d+$")


def parse_tau2_task_id(episode_id: str, domain: str | None) -> str | None:
    """Recover the tau2 task id embedded in the tau-usi instance id.

    tau-usi instance ids are ``{domain}_{tau2_task_id}[_annN]`` (e.g. ``airline_0``,
    ``airline_0_ann2`` -> tau2 airline task ``0``). NOTE: ``metadata.tau_usi_task_id`` is
    an unrelated internal tau-usi id, NOT the tau2 task id, so it must not be used here.
    """

    base = _ANN_SUFFIX.sub("", str(episode_id))  # drop annotation-variant suffix
    if domain and base.startswith(f"{domain}_"):
        return base[len(domain) + 1:] or None
    return base.split("_", 1)[1] if "_" in base else None


def _task_tokens(text: str | None) -> set[str]:
    return set(re.findall(r"[a-z0-9_]+", (text or "").lower()))


# The tau-usi task_description wraps the tau2 task's user instruction in a fixed template
# ("You are role-playing ... agent.\n\n<TASK CORE>\n\n**You may start with:** ..."). Extract the
# core so matching isn't diluted by the shared boilerplate.
_CORE_RE = re.compile(r"agent\.\s*(.*?)\s*\*\*You may start with", re.S | re.I)


def _episode_core_tokens(task_description: str | None) -> set[str]:
    m = _CORE_RE.search(task_description or "")
    return _task_tokens(m.group(1) if m else task_description)


def _tau2_task_cores(repo_path: str | Path, domain: str) -> list[tuple[str, set[str]]]:
    """(task_id, token-set of reason_for_call+known_info+task_instructions) for a tau2 domain."""

    import json

    path = Path(repo_path) / "data" / "tau2" / "domains" / domain / "tasks.json"
    if not path.exists():
        return []
    cores: list[tuple[str, set[str]]] = []
    for task in json.loads(path.read_text(encoding="utf-8")):
        instr = ((task.get("user_scenario") or {}).get("instructions") or {})
        core = " ".join(
            str(instr.get(k) or "") for k in ("reason_for_call", "known_info", "task_instructions")
        )
        cores.append((str(task.get("id")), _task_tokens(core)))
    return cores


def resolve_tau2_task_ids(
    episodes: list[dict[str, Any]], repo_path: str | Path, *, min_score: float = 0.5
) -> dict[str, int]:
    """Overwrite each episode's ``tau2_task_id`` with the CONTENT-matched tau2 task id.

    The numeric suffix of the tau-usi instance id (``airline_0``) does NOT correspond to the
    tau2 task id (``airline_0`` = mia_li booking = tau2 airline task 20, not 0), so matching by
    suffix runs a different task than the recorded conversation. Instead, join each episode to the
    tau2 task whose instruction core has the highest token-Jaccard overlap with the episode's
    task_description core. Fail-closed: episodes below ``min_score`` get ``tau2_task_id=None`` (and
    are skipped downstream). Returns a counts dict for the mapping report.
    """

    cores_by_domain: dict[str, list[tuple[str, set[str]]]] = {}
    counts = {"matched": 0, "unmatched": 0}
    for ep in episodes:
        domain = ep.get("tau2_domain")
        if not domain:
            ep["tau2_task_id"] = None
            counts["unmatched"] += 1
            continue
        cores = cores_by_domain.setdefault(domain, _tau2_task_cores(repo_path, domain))
        etoks = _episode_core_tokens(ep.get("task_description"))
        best_id, best_score = None, 0.0
        for tid, ttoks in cores:
            if not ttoks:
                continue
            inter = len(etoks & ttoks)
            union = len(etoks | ttoks) or 1
            score = inter / union
            if score > best_score:
                best_score, best_id = score, tid
        if best_id is not None and best_score >= min_score:
            ep["tau2_task_id"] = best_id
            counts["matched"] += 1
        else:
            ep["tau2_task_id"] = None
            counts["unmatched"] += 1
    return counts


def load_episodes(normalized_path: str | Path) -> list[dict[str, Any]]:
    episodes = []
    for row in read_jsonl(str(normalized_path)):
        episode_id = str(row.get("episode_id") or row.get("task_id") or "")
        metadata = row.get("metadata") or {}
        tau2_domain = parse_tau2_domain(episode_id, metadata)
        tau2_task_id = parse_tau2_task_id(episode_id, tau2_domain)
        task_description = row.get("task_description") or row.get("intent") or ""
        episodes.append(
            {
                "episode_id": episode_id,
                "tau2_task_id": tau2_task_id,
                "tau2_domain": tau2_domain,
                "real_conversation": row.get("real_conversation") or [],
                "task_description": task_description,
                "intent": row.get("intent") or "",
                "scenario": scenario_only(task_description),
            }
        )
    return episodes


_USER_SIM_INTRO = "You are role-playing as a user interacting with an agent."


def scenario_only(task_description: str) -> str:
    """Extract just the task scenario from a tau-usi ``task_description``.

    Drops the tau2 user-sim wrapper: the leading role-play intro line and everything from the
    first bold section (``**You may start with:**`` / ``**Rules:**`` / ``**Finish the Task:**``
    / ``**Task split:**`` ...) onward, leaving only the concrete task the user wants done (user
    id + all constraints). Verified consistent across all tau-usi episodes (every one starts
    with the intro line and the first bold header is ``You may start with:``)."""

    text = str(task_description or "").strip()
    if text.startswith(_USER_SIM_INTRO):
        text = text[len(_USER_SIM_INTRO):].strip()
    idx = text.find("**")
    if idx != -1:
        text = text[:idx].strip()
    return text


def build_cases(
    episodes: list[dict[str, Any]],
    embeddings_path: str | Path,
    *,
    arms: tuple[str, ...] = CUE_ARMS,
    seed: int = 0,
    require_embeddings: bool = True,
    sampler_checkpoint: str | Path | None = None,
    sampler_seed: int | None = None,
    sampler_guidance_w: float = 1.5,
    sampler_ddim_steps: int | None = None,
    sampler_device: str | None = None,
    sampler_pop_max_k: int | None = None,
    sampler_project_manifold: bool | None = None,
) -> tuple[dict[str, list[dict[str, Any]]], dict[str, int]]:
    """Return ``{arm: [case, ...]}`` and a counts dict for the mapping smoke report.

    CUE / retrieval methods (``require_embeddings=True``) join a per-arm embedding and fail
    closed without one. Persona baselines (usp/realusersim) and base/as_is
    (``require_embeddings=False``) only need a resolvable tau2 task id + domain — the arm
    selects an injected profile/persona, not an embedding, so dummy embeddings are fine and
    ``cue_embedding`` is left None.

    ``sampled_uncond`` / ``sampled_pop`` draw one diffusion sample per resolvable episode
    (matching the paired user count); ``sampled_pop`` conditions each draw on a random
    ``sampler_pop_max_k``-sized subset of that domain's real embeddings.
    """

    from cue_training.evaluation.common.embeddings import load_embeddings, load_raw_embeddings
    from cue_training.evaluation.common.sampler_arms import generate_sampled_arm_embeddings, has_sampled_arms
    from cue_training.evaluation.common.shuffle import build_shuffle_map

    by_key, means = load_embeddings(str(embeddings_path))
    raw_by_key = load_raw_embeddings(str(embeddings_path))

    # Shuffle over every domain-resolvable episode so persona sidecars / CUE share one map,
    # even when embeddings are missing for a subset.
    shuffle_pool = [
        ep for ep in episodes if ep["tau2_task_id"] and ep["tau2_domain"]
    ]
    shuffle_map = build_shuffle_map(
        [ep["episode_id"] for ep in shuffle_pool],
        domains={ep["episode_id"]: ep["tau2_domain"] for ep in shuffle_pool},
        seed=seed,
    )
    # Sampled arms always need real paired embeddings for population + episode count.
    need_emb = bool(require_embeddings or has_sampled_arms(arms))
    resolvable = [
        ep for ep in shuffle_pool
        if (not need_emb or by_key.get(ep["episode_id"]) is not None)
    ]

    sampled_vecs: dict[str, dict[str, list[float]]] = {}
    if has_sampled_arms(arms):
        sampled_vecs = generate_sampled_arm_embeddings(
            episode_ids=[ep["episode_id"] for ep in resolvable],
            domains={ep["episode_id"]: ep["tau2_domain"] for ep in resolvable},
            by_key=raw_by_key,
            arms=arms,
            sampler_checkpoint=sampler_checkpoint or "",
            seed=int(seed if sampler_seed is None else sampler_seed),
            guidance_w=sampler_guidance_w,
            ddim_steps=sampler_ddim_steps,
            device=sampler_device,
            pop_max_k=sampler_pop_max_k,
            project_manifold=sampler_project_manifold,
        )

    cases: dict[str, list[dict[str, Any]]] = {arm: [] for arm in arms}
    for ep in resolvable:
        base = {k: ep[k] for k in ("episode_id", "tau2_task_id", "tau2_domain", "task_description", "real_conversation", "intent", "scenario")}
        base["shuffled_episode_id"] = shuffle_map[ep["episode_id"]]
        for arm in arms:
            if (not require_embeddings and arm not in sampled_vecs) or arm in NO_EMBED_ARMS:
                # Persona/as_is: keep the episode; the arm selects a profile, not an embedding.
                cases[arm].append({**base, "arm": arm, "cue_embedding": None})
                continue
            if arm == "paired":
                emb = by_key.get(ep["episode_id"])
            elif arm == "sample_shuffled":
                emb = by_key.get(shuffle_map[ep["episode_id"]])
            elif arm == "dataset_mean":
                emb = means.get(ep["tau2_domain"])
            elif arm in sampled_vecs:
                emb = sampled_vecs[arm].get(ep["episode_id"])
            else:
                emb = None
            if emb is None:
                continue
            cases[arm].append({**base, "arm": arm, "cue_embedding": emb})

    n_with_emb = sum(1 for e in episodes if by_key.get(e["episode_id"]) is not None)
    counts = {
        "n_total": len(episodes),
        "n_with_tau2_task_id": sum(1 for e in episodes if e["tau2_task_id"]),
        "n_domain_resolved": sum(1 for e in episodes if e["tau2_domain"]),
        "n_embedding_joined": n_with_emb if require_embeddings else len(resolvable),
        "n_skipped": len(episodes) - len(resolvable),
    }
    per_domain = defaultdict(int)
    for ep in resolvable:
        per_domain[ep["tau2_domain"]] += 1
    counts["by_domain"] = dict(per_domain)
    return cases, counts
