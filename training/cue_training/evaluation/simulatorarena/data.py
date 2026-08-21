"""Build SimulatorArena cases + join to CUE embeddings/profiles.

Two domains are supported. Writing (document_creation): each annotation is keyed
``{model}_{workerId}_{document_type}_{intent}``. Math (math_tutoring): keyed
``{model}_{workerId}_{problem_id}``. We join to the CUE (simulatorarena) episode for the
paired CUE embedding, and to the per-conversation extracted profile for the baseline
matched/random arms. Fails closed: unmapped annotations are reported as skipped.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from cue_training.evaluation.common.conditioning import CUE_ARMS as _SHARED_CUE_ARMS

CUE_ARMS = _SHARED_CUE_ARMS
WRITING_DOMAIN = "writing"
MATH_DOMAIN = "math"

# Profile files (HOW) per domain. Math has no "preferences"; it adds "knowledge_state".
_PROFILE_FILES = {
    WRITING_DOMAIN: ("writing_style", "interaction_style", "preferences"),
    MATH_DOMAIN: ("writing_style", "interaction_style", "knowledge_state"),
}

# Shipped GPT-4o profiles under the SimulatorArena submodule (see data/README.md).
_DEFAULT_PROFILE_SUBDIR = {
    WRITING_DOMAIN: "document_creation",
    MATH_DOMAIN: "math_tutoring",
}


def default_profiles_dir(domain: str = WRITING_DOMAIN) -> Path | None:
    """Return the shipped ``user_simulator_profiles/<domain>`` dir if present, else None."""

    from cue_training.evaluation.simulatorarena.bridge import repo_path

    sub = _DEFAULT_PROFILE_SUBDIR.get(domain)
    if not sub:
        return None
    path = repo_path() / "data" / "user_simulator_profiles" / sub
    return path if path.is_dir() else None


def resolve_profiles_dir(
    profiles_dir: str | Path | None, domain: str = WRITING_DOMAIN
) -> Path | None:
    """Use an explicit profiles dir, or fall back to the shipped SimulatorArena profiles."""

    if profiles_dir is not None:
        path = Path(profiles_dir)
        return path if path.is_dir() else None
    return default_profiles_dir(domain)


def annotation_key(ann: dict[str, Any], domain: str = WRITING_DOMAIN) -> str:
    if domain == MATH_DOMAIN:
        return f"{ann.get('model')}_{ann.get('workerId')}_{ann.get('problem_id')}"
    return f"{ann.get('model')}_{ann.get('workerId')}_{ann.get('document_type')}_{ann.get('intent')}"


def _profile_key(ann: dict[str, Any], domain: str = WRITING_DOMAIN) -> str:
    """Key into the profile files. Writing profiles are keyed by the annotation_key; math
    profiles are keyed by ``repr((problem_id, workerId, model))``."""

    if domain == MATH_DOMAIN:
        return repr((ann.get("problem_id"), ann.get("workerId"), ann.get("model")))
    return annotation_key(ann, domain)


def load_annotations(path: str | Path) -> list[dict[str, Any]]:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    return data if isinstance(data, list) else list(data.values())


def _background_text(background: list[dict[str, Any]] | None) -> str:
    lines = []
    for item in background or []:
        lines.append(f"- {item.get('question')}\n    {item.get('answer')}")
    return "\n".join(lines).strip()


def _goal_text(ann: dict[str, Any], background_text: str, domain: str = WRITING_DOMAIN) -> str:
    """WHAT the user wants."""

    if domain == MATH_DOMAIN:
        parts = [f"Math problem: {ann.get('question')}"]
        if ann.get("type"):
            parts.append(f"Topic: {ann.get('type')}")
        if ann.get("level"):
            parts.append(f"Difficulty: {ann.get('level')}")
        if ann.get("math_expertise"):
            parts.append(f"Expertise: {ann.get('math_expertise')}")
        return "\n".join(str(p) for p in parts if p)
    parts = [f"Document type: {ann.get('document_type')}", f"Intent: {ann.get('intent')}"]
    if background_text:
        parts.append(f"\nBackground:\n{background_text}")
    return "\n".join(str(p) for p in parts if p)


_FEATURE_SECTIONS = {
    "preferences": ("## Document Preferences", "Preference Name", "Preference Question Answer"),
    "writing_style": ("## Writing Style", "Feature Name", "Feature Question Answer"),
    "interaction_style": ("## Interaction Style", "Feature Name", "Feature Question Answer"),
}


def _profile_text(
    key: str, profile_files: dict[str, dict[str, Any]], domain: str = WRITING_DOMAIN
) -> str | None:
    """HOW the user behaves: combined style/interaction/(preference|knowledge) persona."""

    out: list[str] = []
    for fname in _PROFILE_FILES.get(domain, _PROFILE_FILES[WRITING_DOMAIN]):
        entry = (profile_files.get(fname) or {}).get(key)
        if not entry:
            continue
        if fname == "knowledge_state":
            # Math knowledge_state is a dict {concepts: [{Concept Name, Status}, ...], ...}.
            concepts = entry.get("concepts") if isinstance(entry, dict) else None
            if not concepts:
                continue
            out.append("## Knowledge State")
            for c in concepts:
                out.append(f"- {c.get('Concept Name')}: {c.get('Status')}")
            continue
        header, name_key, ans_key = _FEATURE_SECTIONS[fname]
        out.append(header)
        for feat in entry:
            out.append(f"- {feat.get(name_key)}: {feat.get(ans_key)}")
    text = "\n".join(out).strip()
    return text or None


def _real_conversation(ann: dict[str, Any]) -> list[dict[str, str]]:
    """Reconstruct the human reference transcript from user_queries + ai_responses."""

    convo: list[dict[str, str]] = []
    users = ann.get("user_queries") or []
    assistants = ann.get("ai_responses") or ann.get("assistant_responses") or []
    for i, u in enumerate(users):
        convo.append({"role": "user", "content": str(u)})
        if i < len(assistants):
            convo.append({"role": "assistant", "content": str(assistants[i])})
    return convo


def _resolve_embedding(
    ann: dict[str, Any], by_key: dict[str, list[float]], domain: str = WRITING_DOMAIN
) -> list[float] | None:
    candidates = [annotation_key(ann, domain), str(ann.get("workerId"))]
    if domain == MATH_DOMAIN:
        candidates.append(str(ann.get("problem_id")))
    else:
        candidates.append(str(ann.get("intent")))
    for candidate in candidates:
        emb = by_key.get(candidate)
        if emb is not None:
            return emb
    return None


def _composite_from_metadata(md: dict[str, Any], domain: str = WRITING_DOMAIN) -> str | None:
    """Rebuild the annotation_key from a CUE embedding row's metadata (the encoder export
    carries workerId as metadata.user_id)."""

    if not md:
        return None
    model, worker = md.get("model"), md.get("user_id") or md.get("workerId")
    if domain == MATH_DOMAIN:
        problem_id = md.get("problem_id")
        if None in (model, worker, problem_id):
            return None
        return f"{model}_{worker}_{problem_id}"
    document_type, intent = md.get("document_type"), md.get("intent")
    if None in (model, worker, document_type, intent):
        return None
    return f"{model}_{worker}_{document_type}_{intent}"


def _load_embeddings_by_composite(path: str | Path, domain: str = WRITING_DOMAIN) -> dict[str, list[float]]:
    """Map each embedding to the annotation composite key via its metadata (the encoder
    export is keyed by per-conversation session ids like ``{workerId}-{index}``, which the
    annotation join key never matches)."""

    import numpy as np

    from cue_training.evaluation.common.embeddings import read_embedding_rows

    rows = read_embedding_rows(str(path))
    out: dict[str, list[float]] = {}
    for row in rows:
        emb = row.get("trajectory_embedding")
        md = row.get("metadata") if isinstance(row.get("metadata"), dict) else {}
        key = _composite_from_metadata(md, domain)
        if emb is None or key is None:
            continue
        out[key] = np.asarray(emb, dtype=np.float32).tolist()
    return out


def _load_json(path: str | Path | None) -> dict[str, Any]:
    if not path or not Path(path).exists():
        return {}
    return json.loads(Path(path).read_text(encoding="utf-8"))


def to_normalized_rows(
    annotations: list[dict[str, Any]], domain: str = WRITING_DOMAIN, background_path: str | Path | None = None
) -> list[dict[str, Any]]:
    """MirrorBench-style normalized rows for the CUE encoder embedding export. The episode_id
    equals the annotation_key so the encoder export joins back per conversation."""

    background = _load_json(background_path)
    rows: list[dict[str, Any]] = []
    for ann in annotations:
        key = annotation_key(ann, domain)
        convo = _real_conversation(ann)
        if not convo:
            continue
        metadata = {
            "domain": domain,
            "dataset": "simulator_arena",
            "model": ann.get("model"),
            "user_id": ann.get("workerId"),
        }
        if domain == MATH_DOMAIN:
            metadata.update({"problem_id": ann.get("problem_id"), "level": ann.get("level"),
                             "type": ann.get("type"), "math_expertise": ann.get("math_expertise")})
        else:
            metadata.update({"document_type": ann.get("document_type"), "intent": ann.get("intent")})
        rows.append({
            "dataset": "simulator_arena",
            "episode_id": key,
            "task_id": key,
            "real_conversation": convo,
            "metadata": metadata,
            "intent": ann.get("intent"),
            "task_description": _goal_text(ann, _background_text(background.get(key)), domain),
        })
    return rows


def build_cases(
    annotations: list[dict[str, Any]],
    embeddings_path: str | Path,
    *,
    arms: tuple[str, ...] = CUE_ARMS,
    background_path: str | Path | None = None,
    profiles_dir: str | Path | None = None,
    seed: int = 0,
    domain: str = WRITING_DOMAIN,
    require_embeddings: bool = True,
    sampler_checkpoint: str | Path | None = None,
    sampler_seed: int | None = None,
    sampler_guidance_w: float = 1.5,
    sampler_ddim_steps: int | None = None,
    sampler_device: str | None = None,
    sampler_pop_max_k: int | None = None,
    sampler_project_manifold: bool | None = None,
) -> tuple[dict[str, list[dict[str, Any]]], dict[str, int]]:
    """Build cases, joining each annotation to its CUE embedding.

    CUE / retrieval variants require the embedding join and fail closed without it.
    Persona / as_is baselines (``require_embeddings=False``) only need the annotation +
    profile; a dummy/empty embeddings file is fine and ``cue_embedding`` is left None.

    ``sampled_uncond`` / ``sampled_pop`` draw one diffusion sample per joined annotation
    (matching the paired user count); ``sampled_pop`` conditions each draw on a random
    ``sampler_pop_max_k``-sized subset of this domain's real embeddings.
    """

    from cue_training.evaluation.common.embeddings import load_embeddings, load_raw_embeddings
    from cue_training.evaluation.common.sampler_arms import generate_sampled_arm_embeddings, has_sampled_arms

    by_key, means = load_embeddings(str(embeddings_path))
    raw_by_key = load_raw_embeddings(str(embeddings_path))
    by_composite = _load_embeddings_by_composite(str(embeddings_path), domain)
    mean_vec = means.get(domain) or (next(iter(means.values()), None))

    background = _load_json(background_path)
    profile_files = {}
    pdir = resolve_profiles_dir(profiles_dir, domain)
    if pdir is not None:
        for name in _PROFILE_FILES.get(domain, _PROFILE_FILES[WRITING_DOMAIN]):
            profile_files[name] = _load_json(pdir / f"{name}.json")

    need_emb = bool(require_embeddings or has_sampled_arms(arms))
    resolved: list[tuple[dict[str, Any], list[float] | None]] = []
    for ann in annotations:
        # Prefer the metadata composite (matches annotation_key); fall back to id candidates.
        emb = by_composite.get(annotation_key(ann, domain)) or _resolve_embedding(ann, by_key, domain)
        if emb is not None or not need_emb:
            resolved.append((ann, emb))

    from cue_training.evaluation.common.shuffle import build_shuffle_map

    # Shuffle over the full annotation pool (including missing embeddings) so persona
    # baselines and CUE share one within-domain map for this domain.
    all_keys = [
        annotation_key(ann, domain)
        for ann in annotations
        if _real_conversation(ann)
    ]
    shuffle_source = build_shuffle_map(
        all_keys,
        domains={key: domain for key in all_keys},
        seed=seed,
    )
    emb_by_key = {
        annotation_key(a, domain): e for a, e in resolved if e is not None
    }
    raw_emb_by_key = {
        key: raw_by_key.get(key, emb) for key, emb in emb_by_key.items()
    }
    shuffle_emb = {
        key: emb_by_key.get(shuffle_source[key])
        for key in all_keys
        if key in shuffle_source
    }

    sampled_vecs: dict[str, dict[str, list[float]]] = {}
    if has_sampled_arms(arms):
        # One sample per joined user (paired resolvable set).
        paired_keys = [annotation_key(a, domain) for a, e in resolved if e is not None]
        sampled_vecs = generate_sampled_arm_embeddings(
            episode_ids=paired_keys,
            domains={key: domain for key in paired_keys},
            by_key=raw_emb_by_key,
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
    n_profile = 0
    n_with_emb = 0
    for ann, emb in resolved:
        key = annotation_key(ann, domain)
        background_text = _background_text(background.get(key))
        n_with_emb += emb is not None
        profile = _profile_text(_profile_key(ann, domain), profile_files, domain)
        n_profile += bool(profile)
        base = {
            "annotation_key": key,
            "episode_id": key,
            "workerId": ann.get("workerId"),
            "assistant_model": ann.get("model"),
            "document_type": ann.get("document_type"),
            "intent": ann.get("intent"),
            "problem_id": ann.get("problem_id"),
            "background": background_text,
            "math_problem": ann.get("question"),
            "task_description": _goal_text(ann, background_text, domain),
            "real_conversation": _real_conversation(ann),
            "profile": profile,
            "shuffled_episode_id": shuffle_source[key],
        }
        for arm in arms:
            if (not require_embeddings and arm not in sampled_vecs):
                # Non-CUE baselines: keep the paired episode set (arm selects profile, not embedding).
                vec = None
            elif arm == "paired":
                vec = emb
            elif arm == "sample_shuffled":
                vec = shuffle_emb.get(key)
            elif arm == "dataset_mean":
                vec = mean_vec
            elif arm in sampled_vecs:
                vec = sampled_vecs[arm].get(key)
            else:
                vec = None
            if require_embeddings or arm in sampled_vecs:
                if vec is None:
                    continue
            cases[arm].append({**base, "arm": arm, "cue_embedding": vec})

    counts = {
        "n_total": len(annotations),
        "n_embedding_joined": n_with_emb,
        "n_profile_joined": n_profile,
        "n_skipped": len(annotations) - len(resolved),
    }
    return cases, counts
