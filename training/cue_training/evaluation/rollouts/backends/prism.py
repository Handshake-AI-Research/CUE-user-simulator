"""PRISM closed-loop normalized-conversation rollout backend."""

from __future__ import annotations

import json
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Mapping, Sequence

from cue_training.evaluation.common.io import append_jsonl, read_jsonl, write_jsonl
from cue_training.evaluation.common.mirrorbench_export import is_empty_rollout_record
from cue_training.evaluation.common.user_sims import (
    SidecarError,
    is_transient_provider_error,
    post_sidecar_json,
)
from cue_training.evaluation.rollouts.backends.base import BackendContext, BenchmarkBackend
from cue_training.evaluation.rollouts.backends.closed_loop import (
    build_episode_sim,
    default_turn_budget,
    opening_prompt,
    prism_user_system_prompt,
    run_closed_loop,
)
from cue_training.evaluation.rollouts.config import ConfigError
from cue_training.evaluation.rollouts.planning import policy_served_name
from cue_training.runlog.log import log, progress, warn


_PRISM_ASSISTANT_SYSTEM = (
    "You are a helpful assistant. Respond helpfully and directly to the user's messages. "
    "Stay on topic and do not invent tool calls."
)
_NO_EMBED_ARMS = frozenset({"as_is"})

EpisodeKey = tuple[str, str, str]


def _read_persona_map(path: Path) -> dict[tuple[str, str, str], Any]:
    """Sidecar rows of {variant, arm, episode_id, persona} → lookup map."""

    out: dict[tuple[str, str, str], Any] = {}
    for row in read_jsonl(str(path)):
        out[(row["variant"], row["arm"], str(row["episode_id"]))] = row.get("persona")
    return out


def _episode_id(row: dict[str, Any]) -> str:
    return str(row.get("episode_id") or row.get("task_id") or "")


def _result_key(row: Mapping[str, Any], *, default_variant: str = "") -> EpisodeKey:
    return (
        str(row.get("variant") or default_variant),
        str(row.get("arm") or ""),
        str(row.get("episode_id") or row.get("task_id") or ""),
    )


def _is_terminal_result(row: Mapping[str, Any]) -> bool:
    """Finished episodes that should not be re-run.

    New rows use ``status``. Legacy success rows have no ``error`` / ``status``.
    """

    status = row.get("status")
    if status == "ok":
        return True
    if status == "error":
        return False
    return "error" not in row


def prepare_resume(out_dir: Path, *, default_variant: str = "") -> set[EpisodeKey]:
    """Compact partial JSONL outputs and return terminal episode keys to skip."""

    results_path = out_dir / "results.jsonl"
    if not results_path.is_file():
        return set()
    rows = read_jsonl(str(results_path))
    terminal = [r for r in rows if _is_terminal_result(r)]
    write_jsonl(str(results_path), terminal)
    done = {_result_key(r, default_variant=default_variant) for r in terminal}
    ok_keys = {
        _result_key(r, default_variant=default_variant)
        for r in terminal
        if r.get("status") == "ok" or ("error" not in r and r.get("status") is None)
    }
    rollout_path = out_dir / "rollout.prism.jsonl"
    if rollout_path.is_file():
        kept: list[dict[str, Any]] = []
        for row in read_jsonl(str(rollout_path)):
            meta = row.get("metadata") if isinstance(row.get("metadata"), dict) else {}
            key = (
                str(meta.get("variant") or default_variant),
                str(meta.get("arm") or ""),
                str(row.get("episode_id") or meta.get("episode_id") or ""),
            )
            if key in ok_keys:
                kept.append(row)
        write_jsonl(str(rollout_path), kept)
    return done

def _row_domain(row: dict[str, Any]) -> str:
    meta = row.get("metadata") if isinstance(row.get("metadata"), dict) else {}
    return str(meta.get("domain") or row.get("dataset") or "diversity")


def join_prism_embeddings(
    rows: list[dict[str, Any]],
    embeddings_path: str | Path | None,
    *,
    arms: list[str],
    require_embeddings: bool,
    seed: int = 0,
    sampler_checkpoint: str | Path | None = None,
    sampler_seed: int | None = None,
    sampler_guidance_w: float = 1.5,
    sampler_ddim_steps: int | None = None,
    sampler_device: str | None = None,
    sampler_pop_max_k: int | None = None,
    sampler_project_manifold: bool | None = None,
    precomputed_sampled: Mapping[str, Mapping[str, Sequence[float]]] | None = None,
) -> tuple[list[tuple[dict[str, Any], str, list[float] | None]], dict[str, int]]:
    """Join normalized PRISM rows to CUE embeddings per arm (tau2/simarena parity).

    Returns ``[(row, arm, cue_embedding), ...]`` plus join counts. When
    ``require_embeddings`` is True, episodes without a joinable vector are skipped for
    embedding arms; ``as_is`` / non-CUE jobs keep ``cue_embedding=None``.
    Pass ``precomputed_sampled`` to reuse vectors from prepare (sample before sidecar).
    """

    from cue_training.evaluation.common.embeddings import load_embeddings, load_raw_embeddings
    from cue_training.evaluation.common.sampler_arms import generate_sampled_arm_embeddings, has_sampled_arms
    from cue_training.evaluation.common.shuffle import build_shuffle_map

    by_key: dict[str, list[float]] = {}
    raw_by_key: dict[str, list[float]] = {}
    means: dict[str, list[float]] = {}
    if embeddings_path is not None and Path(embeddings_path).is_file():
        by_key, means = load_embeddings(str(embeddings_path))
        raw_by_key = load_raw_embeddings(str(embeddings_path))

    resolvable = [row for row in rows if _episode_id(row)]
    shuffle_map = build_shuffle_map(
        [_episode_id(row) for row in resolvable],
        domains={_episode_id(row): _row_domain(row) for row in resolvable},
        seed=seed,
    )
    need_emb = bool(require_embeddings or has_sampled_arms(arms))
    if need_emb:
        resolvable = [
            row for row in resolvable if by_key.get(_episode_id(row)) is not None
        ]

    sampled_vecs: dict[str, dict[str, list[float]]] = {}
    if has_sampled_arms(arms):
        if precomputed_sampled is not None:
            sampled_vecs = {
                arm: {str(eid): list(vec) for eid, vec in (by_eid or {}).items()}
                for arm, by_eid in precomputed_sampled.items()
            }
        else:
            eids = [_episode_id(row) for row in resolvable]
            sampled_vecs = generate_sampled_arm_embeddings(
                episode_ids=eids,
                domains={eid: _row_domain(row) for eid, row in zip(eids, resolvable, strict=True)},
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

    tasks: list[tuple[dict[str, Any], str, list[float] | None]] = []
    for row in resolvable:
        eid = _episode_id(row)
        domain = _row_domain(row)
        task_row = {**row, "shuffled_episode_id": shuffle_map[eid]}
        for arm in arms:
            if (not require_embeddings and arm not in sampled_vecs) or arm in _NO_EMBED_ARMS:
                tasks.append((task_row, arm, None))
                continue
            if arm == "paired":
                emb = by_key.get(eid)
            elif arm == "sample_shuffled":
                emb = by_key.get(shuffle_map[eid])
            elif arm == "dataset_mean":
                emb = means.get(domain) or (next(iter(means.values()), None) if means else None)
            elif arm in sampled_vecs:
                emb = sampled_vecs[arm].get(eid)
            else:
                emb = None
            if emb is None:
                continue
            tasks.append((task_row, arm, emb))

    counts = {
        "n_total": len(rows),
        "n_resolvable": len(resolvable),
        "n_embedding_keys": len(by_key),
        "n_tasks": len(tasks),
        "n_skipped_rows": len(rows) - len(resolvable),
        "shuffle_map": shuffle_map,
    }
    return tasks, counts


class PrismBackend(BenchmarkBackend):
    name = "prism"

    def prepare(self, ctx: BackendContext) -> dict[str, Any]:
        pcfg = ctx.cfg.benchmarks.prism
        if not pcfg.enabled:
            raise ConfigError("prism benchmark disabled in config")
        normalized = pcfg.normalized
        if normalized is None:
            raise ConfigError("prism.normalized unset")
        if not Path(normalized).is_file():
            if ctx.dry_run:
                return {
                    "normalized": str(normalized),
                    "embeddings": None,
                    "n_rows": 0,
                    "max_user_turns_cap": pcfg.max_user_turns,
                    "rows": [],
                }
            raise ConfigError(f"prism.normalized missing: {normalized}")

        embeddings = pcfg.embeddings
        if ctx.job.needs_cue_embeddings and (embeddings is None or not Path(embeddings).is_file()):
            if ctx.dry_run:
                embeddings = None
            else:
                raise ConfigError(
                    f"prism.embeddings required for {ctx.job.method}: {embeddings}"
                )
        if not ctx.job.needs_cue_embeddings:
            embeddings = None

        rows = read_jsonl(str(normalized))
        if ctx.limit is not None:
            rows = rows[: ctx.limit]

        sampled_path = None
        from cue_training.evaluation.common.sampler_arms import has_sampled_arms

        arms = list(ctx.job.arms or ["as_is"])
        if (
            not ctx.dry_run
            and has_sampled_arms(arms)
            and embeddings is not None
            and Path(embeddings).is_file()
        ):
            # Diffusion sample before the decoder sidecar starts (orchestrator order).
            from cue_training.evaluation.common.embeddings import load_raw_embeddings
            from cue_training.evaluation.common.sampler_arms import generate_sampled_arm_embeddings

            cue = ctx.cfg.cue
            eids = [_episode_id(row) for row in rows if _episode_id(row)]
            raw = load_raw_embeddings(str(embeddings))
            eids = [eid for eid in eids if eid in raw]
            log(
                "rollouts.prism",
                f"pre-sampling {len(eids)} embeddings for {[a for a in arms if a in ('sampled_uncond', 'sampled_pop')]}",
            )
            sampled = generate_sampled_arm_embeddings(
                episode_ids=eids,
                domains={_episode_id(row): _row_domain(row) for row in rows if _episode_id(row)},
                by_key=raw,
                arms=arms,
                sampler_checkpoint=cue.sampler_checkpoint or "",
                seed=int(cue.sampler_seed),
                guidance_w=cue.sampler_guidance_w,
                ddim_steps=cue.sampler_ddim_steps,
                device=cue.sampler_device,
                pop_max_k=cue.sampler_pop_max_k,
                project_manifold=cue.sampler_project_manifold,
            )
            sampled_path = ctx.scratch_dir / "sampled_arm_embeddings.json"
            sampled_path.parent.mkdir(parents=True, exist_ok=True)
            sampled_path.write_text(json.dumps(sampled), encoding="utf-8")
            log("rollouts.prism", f"wrote sampled embeddings -> {sampled_path}")

        return {
            "normalized": str(normalized),
            "embeddings": str(embeddings) if embeddings else None,
            "n_rows": len(rows),
            "max_user_turns_cap": pcfg.max_user_turns,
            "rows": rows if ctx.dry_run else None,
            "sampled_embeddings": str(sampled_path) if sampled_path else None,
        }

    def _assistant_fn(self, ctx: BackendContext, tasks: list[tuple[dict[str, Any], str, Any]] | None = None):
        from cue_training.baselines.common.llm import chat
        from cue_training.evaluation.common.dataset_assistants import (
            DatasetAssistantRouter,
            api_key_env_for_model,
            resolve_prism_assistant,
        )

        match_dataset = bool(ctx.cfg.assistant.match_dataset)
        default_model = ctx.cfg.assistant.model
        default_api_key_env = ctx.cfg.assistant.api_key_env
        api_base = ctx.cfg.assistant.api_base
        max_tokens = ctx.cfg.assistant.sampling.max_tokens
        temperature = ctx.cfg.assistant.sampling.temperature

        def dataset_model(row: dict[str, Any]) -> str:
            meta = row.get("metadata") or {}
            raw = meta.get("assistant_model") or meta.get("assistant_model_name")
            return (
                str(meta["assistant_model_litellm"])
                if meta.get("assistant_model_litellm")
                else resolve_prism_assistant(raw if isinstance(raw, str) else None)
            )

        router = (
            DatasetAssistantRouter(dataset_model(row) for row, _, _ in tasks)
            if match_dataset and tasks
            else None
        )

        def reply_for(row: dict[str, Any]):
            model = dataset_model(row) if match_dataset else default_model

            def call(chosen: str, messages: list[dict[str, str]]) -> str:
                return chat(
                    messages,
                    model=chosen,
                    api_key_env=api_key_env_for_model(chosen) or default_api_key_env,
                    api_base=api_base,
                    max_tokens=max_tokens,
                    temperature=temperature,
                )

            def reply(conversation: list[dict[str, str]], system: str) -> str:
                messages = [{"role": "system", "content": system}, *conversation]
                chosen = router.route(model) if router else model
                try:
                    return call(chosen, messages)
                except Exception:  # noqa: BLE001
                    alt = router.mark_dead(chosen) if router else None
                    if alt is None:
                        raise
                    return call(alt, messages)

            return reply

        return reply_for

    def _sim_extra(self, ctx: BackendContext) -> dict[str, Any]:
        sim = ctx.cfg.simulator(ctx.job.simulator)
        extra: dict[str, Any] = {
            "sim_model": sim.model if sim.kind == "api" else f"hosted_vllm/{sim.served_name or sim.model}",
            "sim_api_key_env": sim.api_key_env,
            "sim_max_tokens": int(sim.sampling.max_tokens),
            "sim_temperature": float(sim.sampling.temperature),
            "sim_top_p": float(sim.sampling.top_p),
            "device": "cuda",
            "dtype": "bfloat16",
            # Baseline registry name comes from the variant (e.g. baseline:realusersim),
            # not job.method — cue/proposals uses method=proposals with that same variant.
            "name": (
                ctx.job.variant.split(":", 1)[1]
                if ctx.job.variant.startswith("baseline:")
                else None
            ),
            "output_dir": str(ctx.cfg.baselines.artifact_root),
            "artifacts_dir": str(ctx.cfg.baselines.artifact_root / ctx.job.method)
            if ctx.job.rollout_type == "baseline"
            else None,
        }
        if sim.kind == "vllm":
            api_base = sim.api_base or (ctx.vllm_urls or {}).get(f"sim:{sim.name}") or f"http://127.0.0.1:{sim.port}/v1"
            extra["sim_api_base"] = api_base
            extra["vllm_base_url"] = api_base
            extra["hf_path"] = sim.model
        elif sim.api_base:
            # Shared external vLLM (kind=api + api_base); see tau2/simarena backends.
            extra["sim_api_base"] = sim.api_base
        if ctx.sidecar_url:
            extra["sidecar_url"] = ctx.sidecar_url
        if ctx.job.rollout_type == "cue":
            ddir = ctx.cfg.cue.decoder_dir_for(ctx.job.method, ctx.job.simulator)
            if ddir:
                extra["decoder_dir"] = str(ddir)
                extra["semantic_model"] = ctx.cfg.cue.semantic_model
                extra["decode_temperature"] = ctx.cfg.cue.decode_temperature
                extra["decode_top_p"] = ctx.cfg.cue.decode_top_p
                extra["decode_num_candidates"] = ctx.cfg.cue.decode_num_candidates
                extra["decode_mode"] = ctx.cfg.cue.decode_mode
                extra["decode_slot_dedup_jaccard"] = ctx.cfg.cue.decode_slot_dedup_jaccard
                extra["decode_noop_retries"] = ctx.cfg.cue.decode_noop_retries
                extra["decode_noop_temperature"] = ctx.cfg.cue.decode_noop_temperature
                extra["decode_batch_size"] = ctx.cfg.cue.decode_batch_size
                extra["example_retrieval_enabled"] = ctx.cfg.cue.example_retrieval_enabled
                extra["example_retrieval_k_sessions"] = ctx.cfg.cue.example_retrieval_k_sessions
                extra["example_retrieval_n_general"] = ctx.cfg.cue.example_retrieval_n_general
                extra["example_retrieval_n_specific"] = ctx.cfg.cue.example_retrieval_n_specific
                extra["encode_session_preprocess"] = ctx.cfg.cue.session_preprocess_for(
                    ctx.job.domain
                )
        urls = ctx.vllm_urls or {}
        for key in ("userlm", "usp"):
            if key in urls:
                extra[f"{key}_vllm_base_url"] = urls[key]
                extra[f"{key}_vllm_model"] = {
                    "userlm": "userlm",
                    "usp": policy_served_name(ctx.cfg, "usp"),
                }[key]
        return {k: v for k, v in extra.items() if v is not None}

    def _predecode_manuals(
        self,
        ctx: BackendContext,
        tasks: list[tuple[dict[str, Any], str, list[float] | None]],
        extra: dict[str, Any],
    ) -> dict[str, str]:
        """Batch-decode manuals for decoder jobs (tau2/simarena parity).

        Prism is ~8k episodes × arms; one giant /decode_manuals POST hangs the client and
        the sidecar, so we chunk (and log) instead of sending the full set at once.
        """

        if ctx.job.variant != "decoder":
            return {}
        decoder_dir = extra.get("decoder_dir")
        if not decoder_dir:
            warn(
                "rollouts.prism",
                "decoder job has no decoder_dir — skipping pre-decode; on-demand decode "
                "under high concurrency will look hung",
            )
            return {}
        live_encode = not ctx.job.needs_cue_embeddings
        items: list[tuple[str, list[float] | list[dict[str, Any]]]] = []
        if live_encode:
            by_episode = {_episode_id(row): row for row, _, _ in tasks}
            for row, arm, _ in tasks:
                source_id = (
                    str(row.get("shuffled_episode_id") or _episode_id(row))
                    if arm == "sample_shuffled"
                    else _episode_id(row)
                )
                source = by_episode.get(source_id, row)
                trajectory = list(source.get("real_conversation") or [])
                if trajectory:
                    items.append((f"{arm}::{_episode_id(row)}", trajectory))
        else:
            items = [
                (f"{arm}::{_episode_id(row)}", embedding)
                for row, arm, embedding in tasks
                if embedding is not None
            ]
        seen: set[str] = set()
        unique: list[tuple[str, list[float] | list[dict[str, Any]]]] = []
        for key, value in items:
            if key in seen:
                continue
            seen.add(key)
            unique.append((key, value))
        if not unique:
            return {}

        max_new = 1024
        temp = float(extra.get("decode_temperature") or 0.0)
        top_p = float(extra.get("decode_top_p") or 1.0)
        n_cand = int(extra.get("decode_num_candidates") or 1)
        mode = str(extra.get("decode_mode") or "sample")
        bsz = int(extra.get("decode_batch_size") or 16)
        retrieval_enabled = bool(extra.get("example_retrieval_enabled"))
        # Retrieval runs after candidate selection; do not force single-candidate sample.
        # Keep HTTP payloads / GPU work bounded (natural×N×dual on 16k is multi-hour).
        chunk_size = max(bsz, min(512, len(unique)))
        n_chunks = (len(unique) + chunk_size - 1) // chunk_size
        log(
            "rollouts.prism",
            f"pre-decoding {len(unique)} manuals "
            f"(mode={mode}, candidates={n_cand}, chunks={n_chunks}×{chunk_size}) "
            f"— this can take a while before the episode bar moves",
        )

        out: dict[str, str] = {}
        for ci in range(n_chunks):
            chunk = unique[ci * chunk_size : (ci + 1) * chunk_size]
            log(
                "rollouts.prism",
                f"pre-decode chunk {ci + 1}/{n_chunks} ({len(chunk)} manuals)",
            )
            if ctx.sidecar_url:
                payload = {
                    "decoder_dir": decoder_dir,
                    "items": [
                        {
                            "key": key,
                            "trajectory" if live_encode else "cue_embedding": value,
                        }
                        for key, value in chunk
                    ],
                    "max_new_tokens": max_new,
                    "temperature": temp,
                    "top_p": top_p,
                    "num_candidates": n_cand,
                    "decode_mode": mode,
                    "decode_slot_dedup_jaccard": float(
                        extra.get("decode_slot_dedup_jaccard") or 0.5
                    ),
                    "decode_noop_retries": int(extra.get("decode_noop_retries") or 0),
                    "decode_noop_temperature": float(
                        extra.get("decode_noop_temperature") or 0.7
                    ),
                    "batch_size": bsz,
                    "example_retrieval_enabled": retrieval_enabled,
                    "example_retrieval_k_sessions": int(extra.get("example_retrieval_k_sessions") or 8),
                    "example_retrieval_n_general": int(extra.get("example_retrieval_n_general") or 2),
                    "example_retrieval_n_specific": int(extra.get("example_retrieval_n_specific") or 2),
                    "session_preprocess": str(
                        extra.get("encode_session_preprocess") or "full"
                    ),
                }
                # Natural/diverse on a few hundred dual manuals can exceed an hour on a busy GPU.
                body = post_sidecar_json(
                    f"{str(ctx.sidecar_url).rstrip('/')}/decode_manuals",
                    payload,
                    timeout=7200,
                    tag="rollouts.prism",
                )
                if body.get("error"):
                    raise SidecarError(
                        f"CUE pre-decode failed for prism chunk {ci + 1}/{n_chunks}: {body['error']}"
                    )
                part = {str(k): str(v) for k, v in (body.get("manuals") or {}).items()}
            else:
                from cue_training.evaluation.common.user_sims import decode_manuals_batch

                part = decode_manuals_batch(
                    decoder_dir=str(decoder_dir),
                    items=chunk,
                    device=str(extra.get("device") or "cuda"),
                    dtype=str(extra.get("dtype") or "bfloat16"),
                    max_new_tokens=max_new,
                    temperature=temp,
                    top_p=top_p,
                    num_candidates=n_cand,
                    decode_mode=mode,
                    decode_slot_dedup_jaccard=float(
                        extra.get("decode_slot_dedup_jaccard") or 0.5
                    ),
                    decode_noop_retries=int(extra.get("decode_noop_retries") or 0),
                    decode_noop_temperature=float(
                        extra.get("decode_noop_temperature") or 0.7
                    ),
                    batch_size=bsz,
                    example_retrieval_enabled=retrieval_enabled,
                    example_retrieval_k_sessions=int(extra.get("example_retrieval_k_sessions") or 8),
                    example_retrieval_n_general=int(extra.get("example_retrieval_n_general") or 2),
                    example_retrieval_n_specific=int(extra.get("example_retrieval_n_specific") or 2),
                    session_preprocess=str(
                        extra.get("encode_session_preprocess") or "full"
                    ),
                )
            empty = [k for k, v in part.items() if not str(v).strip()]
            if len(part) != len(chunk) or empty:
                raise SidecarError(
                    f"CUE pre-decode chunk {ci + 1}/{n_chunks} returned "
                    f"{len(part) - len(empty)}/{len(chunk)} usable manuals; empty={empty[:3]}"
                )
            out.update(part)

        log("rollouts.prism", f"pre-decoded {len(out)} manuals")
        return out

    def run(self, ctx: BackendContext, prep: dict[str, Any]) -> dict[str, Any]:
        replace_arms = list(ctx.replace_arms or [])
        # Stage arm-subset reruns so sibling arms in job_dir are not overwritten.
        out_dir = (ctx.scratch_dir / ".arm_rerun") if replace_arms else ctx.job_dir
        out_dir.mkdir(parents=True, exist_ok=True)
        shards_dir = ctx.scratch_dir / "shards"
        shards_dir.mkdir(parents=True, exist_ok=True)

        arms = list(ctx.job.arms or ["as_is"])
        if ctx.dry_run:
            rows = prep.get("rows") or []
            log(
                "rollouts.prism",
                f"dry-run prism method={ctx.job.method} sim={ctx.job.simulator} "
                f"n={prep.get('n_rows')} arms={arms}",
            )
            return {
                "prep": {k: v for k, v in prep.items() if k != "rows"},
                "shards": [],
                "n_planned": len(rows) * max(1, len(arms)),
            }

        rows = read_jsonl(prep["normalized"])
        if ctx.limit is not None:
            rows = rows[: ctx.limit]
        from cue_training.evaluation.common.sampler_arms import has_sampled_arms

        require = bool(ctx.job.needs_cue_embeddings) or has_sampled_arms(arms)
        cue = ctx.cfg.cue
        precomputed = None
        sampled_path = prep.get("sampled_embeddings")
        if sampled_path and Path(sampled_path).is_file():
            precomputed = json.loads(Path(sampled_path).read_text(encoding="utf-8"))
        tasks, counts = join_prism_embeddings(
            rows,
            prep.get("embeddings"),
            arms=arms,
            require_embeddings=require,
            seed=cue.sampler_seed,
            sampler_checkpoint=cue.sampler_checkpoint,
            sampler_seed=cue.sampler_seed,
            sampler_guidance_w=cue.sampler_guidance_w,
            sampler_ddim_steps=cue.sampler_ddim_steps,
            sampler_device=cue.sampler_device,
            sampler_pop_max_k=cue.sampler_pop_max_k,
            sampler_project_manifold=cue.sampler_project_manifold,
            precomputed_sampled=precomputed,
        )
        shuffle_map: dict[str, str] = counts.pop("shuffle_map", {})  # type: ignore[assignment]
        log(
            "rollouts.prism",
            f"embedding join: total={counts['n_total']} resolvable={counts['n_resolvable']} "
            f"keys={counts['n_embedding_keys']} tasks={counts['n_tasks']} "
            f"skipped_rows={counts['n_skipped_rows']} require={require}",
        )
        if require and counts["n_tasks"] == 0:
            raise ConfigError(
                "prism: no episodes joined to cue embeddings; re-export prism embeddings "
                f"for this checkpoint (config embeddings={prep.get('embeddings')!r})"
            )

        assistant_for = self._assistant_fn(ctx, tasks)
        extra = self._sim_extra(ctx)
        manuals = self._predecode_manuals(ctx, tasks, extra)
        personas_path = ctx.job_dir / "personas.jsonl"
        persona_map = _read_persona_map(personas_path) if personas_path.is_file() else {}
        if persona_map:
            log("rollouts.prism", f"loaded {len(persona_map)} persona rows from {personas_path}")

        results_path = out_dir / "results.jsonl"
        rollout_path = out_dir / "rollout.prism.jsonl"
        done = prepare_resume(out_dir, default_variant=str(ctx.job.variant))
        if done:
            log("rollouts.prism", f"resume: skipping {len(done)} terminal episode(s) already on disk")
        write_lock = threading.Lock()

        def _append_result(row: dict[str, Any]) -> None:
            with write_lock:
                append_jsonl(str(results_path), [row])

        def _append_rollout(row: dict[str, Any]) -> None:
            with write_lock:
                append_jsonl(str(rollout_path), [row])

        def _one(
            row: dict[str, Any], arm: str, cue_embedding: list[float] | None
        ) -> tuple[dict[str, Any], dict[str, Any]]:
            budget = default_turn_budget(row, cap=prep.get("max_user_turns_cap"))
            eid = _episode_id(row)
            persona = persona_map.get((ctx.job.variant, arm, eid))
            ep_extra = dict(extra)
            key = f"{arm}::{eid}"
            if key in manuals:
                ep_extra["command_block"] = manuals[key]
            sim = build_episode_sim(
                variant=ctx.job.variant,
                arm=arm,
                row=row,
                persona=persona,
                cue_embedding=cue_embedding,
                extra=ep_extra,
                system_prompt=prism_user_system_prompt(row),
            )
            origin = eid if arm == "paired" else shuffle_map.get(eid)
            provenance = {
                "variant": ctx.job.variant,
                "arm": arm,
                "method": ctx.job.method,
                "simulator": ctx.job.simulator,
                "source_harness": "rollouts.prism",
                "max_user_turns": budget,
                "persona_origin_episode_id": origin,
                "shuffled_episode_id": shuffle_map.get(eid),
                "assistant_model": (row.get("metadata") or {}).get("assistant_model"),
                "assistant_model_litellm": (row.get("metadata") or {}).get("assistant_model_litellm"),
            }
            record = run_closed_loop(
                row=row,
                user_sim=sim,
                assistant_reply=assistant_for(row),
                assistant_system=_PRISM_ASSISTANT_SYSTEM,
                max_user_turns=budget,
                provenance=provenance,
                opening=opening_prompt(row),
            )
            if is_empty_rollout_record(record):
                raise RuntimeError("empty_user_sim_response")
            result = {
                "episode_id": row.get("episode_id"),
                "task_id": row.get("task_id"),
                "variant": ctx.job.variant,
                "arm": arm,
                "status": "ok",
                "n_user_turns": sum(
                    1 for t in record.get("rollout_conversation") or [] if t.get("role") == "user"
                ),
                "task_success": None,
            }
            return result, record

        pending = [
            (row, arm, emb)
            for row, arm, emb in tasks
            if _result_key(
                {"variant": ctx.job.variant, "arm": arm, "episode_id": _episode_id(row)}
            )
            not in done
        ]
        workers = max(1, int(ctx.cfg.resources.concurrency))
        log(
            "rollouts.prism",
            f"running {len(pending)} episodes "
            f"(skipped {len(tasks) - len(pending)} resumed; "
            f"{counts['n_resolvable']} rows x arms, workers={workers}) "
            f"method={ctx.job.method} sim={ctx.job.simulator}",
        )
        n_ok = 0
        n_err = 0
        if not pending:
            results = read_jsonl(str(results_path)) if results_path.is_file() else []
            rollouts = read_jsonl(str(rollout_path)) if rollout_path.is_file() else []
        else:
            bar = progress(
                total=len(pending),
                desc=f"prism:{ctx.job.method}/{ctx.job.simulator}",
                unit="ep",
                tag="rollouts.prism",
                leave=True,
            )
            try:
                if workers <= 1:
                    for row, arm, emb in pending:
                        try:
                            result, record = _one(row, arm, emb)
                            _append_result(result)
                            _append_rollout(record)
                            done.add(_result_key(result))
                            n_ok += 1
                        except Exception as exc:  # noqa: BLE001
                            # Provider blips are recorded per-episode; a real decoder/sidecar
                            # failure still aborts the run.
                            if isinstance(exc, SidecarError) and not is_transient_provider_error(exc):
                                raise
                            warn("rollouts.prism", f"episode failed: {exc}")
                            _append_result({
                                "episode_id": row.get("episode_id"),
                                "task_id": row.get("task_id"),
                                "variant": ctx.job.variant,
                                "arm": arm,
                                "status": "error",
                                "error": repr(exc),
                            })
                            n_err += 1
                        bar.update(1)
                        bar.set_postfix(ok=n_ok, err=n_err)
                else:
                    with ThreadPoolExecutor(max_workers=workers) as pool:
                        futs = {
                            pool.submit(_one, row, arm, emb): (row, arm)
                            for row, arm, emb in pending
                        }
                        for fut in as_completed(futs):
                            row, arm = futs[fut]
                            try:
                                result, record = fut.result()
                                _append_result(result)
                                _append_rollout(record)
                                done.add(_result_key(result))
                                n_ok += 1
                            except Exception as exc:  # noqa: BLE001
                                # Provider blips are recorded per-episode; a real decoder/sidecar
                                # failure still aborts the run.
                                if isinstance(exc, SidecarError) and not is_transient_provider_error(
                                    exc
                                ):
                                    raise
                                warn("rollouts.prism", f"episode failed: {exc}")
                                _append_result({
                                    "episode_id": row.get("episode_id"),
                                    "task_id": row.get("task_id"),
                                    "variant": ctx.job.variant,
                                    "arm": arm,
                                    "status": "error",
                                    "error": repr(exc),
                                })
                                n_err += 1
                            bar.update(1)
                            bar.set_postfix(ok=n_ok, err=n_err)
            finally:
                bar.close()
            results = read_jsonl(str(results_path)) if results_path.is_file() else []
            rollouts = read_jsonl(str(rollout_path)) if rollout_path.is_file() else []

        log(
            "rollouts.prism",
            f"finished {len(results)} results ({n_ok} ok this run, {n_err} err this run) "
            f"-> {rollout_path}",
        )
        shard_name = "shard_000.jsonl"
        write_jsonl(str(shards_dir / shard_name), results)
        if replace_arms:
            from cue_training.evaluation.rollouts.artifacts import promote_final_artifacts

            promote_final_artifacts(
                out_dir,
                ctx.job_dir,
                ["results.jsonl", "rollout.prism.jsonl"],
                replace_arms=replace_arms,
            )
        (ctx.job_dir / "summary.json").write_text(
            json.dumps(
                {
                    "n_results": len(results) if not replace_arms else None,
                    "n_rollouts": len(rollouts) if not replace_arms else None,
                    "embedding_join": counts,
                    "replaced_arms": replace_arms or None,
                },
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )
        return {"prep": {k: v for k, v in prep.items() if k != "rows"}, "shards": [shard_name]}
