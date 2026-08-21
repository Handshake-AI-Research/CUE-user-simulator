"""Run SimulatorArena doc-creation with our user sims; collect metrics + MirrorBench rollouts.

Produces (a) a standalone results/summary, (b) MirrorBench rollouts by default, and (c)
a SimulatorArena-format conversation JSON that its terminate + evaluation scripts can
consume (schema reconciled against the submodule during execution).
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import threading
import traceback
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

from cue_training.utils.config import storage_root
from cue_training.evaluation.common.io import append_jsonl, read_jsonl, write_jsonl
from cue_training.evaluation.common.conditioning import TurnContext
from cue_training.evaluation.common.mirrorbench_export import has_nonempty_user_turn, to_mirrorbench_rollout
from cue_training.evaluation.common.user_sims import (
    SidecarError,
    build_user_sim,
    is_transient_provider_error,
    post_sidecar_json,
)
from cue_training.evaluation.simulatorarena import bridge
from cue_training.evaluation.simulatorarena.data import build_cases, load_annotations
from cue_training.runlog.log import log, progress, warn
from cue_training.evaluation.simulatorarena.loop import simulate_case

CUE_VARIANTS = {"decoder"}
_DATASET_ASSISTANT_TOKENS = {"", "dataset", "match", "match_dataset"}
PERSONA_BASELINES = {"usp", "realusersim_paired_noex"}  # paper realusersim is as_is WildChat-pool sampling
PAIRED_REALUSERSIM = {"realusersim_paired_noex"}

_DOC_TYPE_MAP = {"blog post": "Blog Post", "email": "Email/Letter", "creative writing": "Creative Writing"}

EpisodeKey = tuple[str, str, str]


def _episode_key(variant: str, arm: str, episode_id: Any) -> EpisodeKey:
    return (str(variant), str(arm), str(episode_id))


def _is_terminal_result(row: dict[str, Any]) -> bool:
    """Episodes that should not be re-run on resume.

    ``ok`` rows are finished. Content-policy blocks are intentionally skipped and
    would only fail the same way again. Transient / empty-sim errors stay
    non-terminal so resume retries them.
    """

    if row.get("status") == "ok":
        return True
    return bool(row.get("content_policy_blocked"))


def prepare_resume(out_dir: Path, *, keep_rollouts: bool) -> set[EpisodeKey]:
    """Compact partial JSONL outputs and return terminal episode keys to skip."""

    results_path = out_dir / "results.jsonl"
    if not results_path.is_file():
        return set()
    rows = read_jsonl(str(results_path))
    terminal = [r for r in rows if _is_terminal_result(r)]
    write_jsonl(str(results_path), terminal)
    done = {
        _episode_key(r["variant"], r["arm"], r["episode_id"])
        for r in terminal
        if r.get("variant") is not None and r.get("arm") is not None
    }
    ok_keys = {
        _episode_key(r["variant"], r["arm"], r["episode_id"])
        for r in terminal
        if r.get("status") == "ok"
    }
    rollout_path = out_dir / "rollout.simulatorarena.jsonl"
    if keep_rollouts and rollout_path.is_file():
        kept: list[dict[str, Any]] = []
        for row in read_jsonl(str(rollout_path)):
            meta = row.get("metadata") if isinstance(row.get("metadata"), dict) else {}
            eid = row.get("episode_id") or meta.get("episode_id")
            key = _episode_key(meta.get("variant"), meta.get("arm"), eid)
            if key in ok_keys:
                kept.append(row)
        write_jsonl(str(rollout_path), kept)
    return done


def _rebuild_native_from_rollouts(out_dir: Path, domain: str) -> None:
    """Rebuild SimulatorArena native JSON from streamed MirrorBench rollouts."""

    path = out_dir / "rollout.simulatorarena.jsonl"
    if not path.is_file():
        return
    native: dict[str, dict[str, Any]] = {}
    for row in read_jsonl(str(path)):
        meta = row.get("metadata") if isinstance(row.get("metadata"), dict) else {}
        variant = meta.get("variant")
        arm = meta.get("arm")
        if not variant or not arm:
            continue
        case = {
            "assistant_model": meta.get("assistant_model"),
            "workerId": meta.get("workerId"),
            "document_type": meta.get("document_type"),
            "intent": meta.get("intent"),
            "problem_id": meta.get("problem_id"),
        }
        conversation = row.get("rollout_conversation") or []
        if not isinstance(conversation, list):
            continue
        _accumulate_native(native, str(variant), str(arm), case, conversation, domain)
    _write_native(out_dir, native)


def _is_content_policy_error(exc: BaseException) -> bool:
    text = repr(exc).lower()
    return "contentpolicyviolation" in text or "prompt was flagged" in text


def _baseline_name(variant: str) -> str:
    return variant.split(":", 1)[1] if variant.startswith("baseline:") else variant


def _uses_cue_cases(variant: str) -> bool:
    return variant in CUE_VARIANTS


def _arms_for(variant: str, requested: list[str]) -> list[str]:
    if variant in CUE_VARIANTS:
        return [
            a for a in requested
            if a in ("paired", "sample_shuffled", "dataset_mean", "sampled_uncond", "sampled_pop")
        ] or ["paired"]
    name = _baseline_name(variant)
    if name in PERSONA_BASELINES:
        if name in PAIRED_REALUSERSIM:
            allowed = ("paired", "sample_shuffled")
            return [a for a in requested if a in allowed] or ["paired"]
        # USP defaults: paired + sample_diverse (sample_shuffled still allowed if requested).
        allowed = ("paired", "sample_shuffled", "sample_diverse")
        return [a for a in requested if a in allowed] or ["paired", "sample_diverse"]
    return ["as_is"]


def _sim_sampling_extra(args: argparse.Namespace) -> dict[str, Any]:
    """Forward simulator generation knobs into user-sim ``extra`` (defaults match prior hardcoded)."""

    return {
        "sim_max_tokens": int(getattr(args, "sim_max_tokens", 512)),
        "sim_temperature": float(getattr(args, "sim_temperature", 0.7)),
        "sim_top_p": float(getattr(args, "sim_top_p", 0.95)),
    }


def _variant_extra(variant: str, args: argparse.Namespace) -> dict[str, Any]:
    common = {"device": args.device, "dtype": args.dtype, **_sim_sampling_extra(args)}
    if variant == "decoder":
        from cue_training.evaluation.common.user_sims import example_retrieval_kwargs

        return {**common, "decoder_dir": args.decoder_dir, "semantic_model": args.semantic_model,
                "sim_model": args.sim_model, "sim_api_base": args.sim_api_base,
                "sim_api_key_env": args.sim_api_key_env, "decode_temperature": args.decode_temperature,
                "decode_top_p": args.decode_top_p,
                "decode_num_candidates": getattr(args, "decode_num_candidates", 1),
                "decode_mode": getattr(args, "decode", "sample"),
                "decode_slot_dedup_jaccard": getattr(args, "decode_slot_dedup_jaccard", 0.5),
                "decode_noop_retries": getattr(args, "decode_noop_retries", 0),
                "decode_noop_temperature": getattr(args, "decode_noop_temperature", 0.7),
                **example_retrieval_kwargs(args)}
    if variant.startswith("baseline:"):
        extra = {**common, "name": variant.split(":", 1)[1], "output_dir": args.baseline_output_dir,
                 "artifacts_dir": args.baseline_artifacts_dir, "sim_model": args.sim_model,
                 "sim_api_base": args.sim_api_base, "sim_api_key_env": args.sim_api_key_env}
        for key in (
            "userlm_vllm_base_url", "userlm_vllm_model", "userlm_vllm_api_key_env",
            "usp_vllm_base_url", "usp_vllm_model", "usp_vllm_api_key_env",
        ):
            val = getattr(args, key, None)
            if val:
                extra[key] = val
        return extra
    if variant == "base_api":
        # Must forward the simulator's key env (e.g. GOOGLE_API_KEY for gemini/*);
        # otherwise BaseModelUserSim defaults to OPENAI_API_KEY and every turn fails
        # closed as empty -> ###STOP###.
        return {
            "sim_model": args.base_api_model,
            "sim_api_base": args.sim_api_base,
            "sim_api_key_env": args.sim_api_key_env,
            **_sim_sampling_extra(args),
        }
    if variant == "base_local":
        extra = {**common, "hf_path": args.base_local_path}
        if getattr(args, "vllm_base_url", None):
            extra["vllm_base_url"] = args.vllm_base_url
        return extra
    return common


def _predecode_manuals(
    args: argparse.Namespace, arm: str, cases: list[dict[str, Any]]
) -> dict[str, str]:
    """Batch-decode this arm's persona manuals up front (sidecar if used, else in-process).

    A few greedy live-encode decodes can return empty strings (unparseable / truncated
    generations). Those keys are retried with stochastic multi-candidate decode; any that
    remain empty are dropped so the arm can continue on the usable majority.
    """

    live_encode = bool(getattr(args, "live_encode_trajectories", False))
    if live_encode:
        by_episode = {str(c["episode_id"]): c for c in cases}
        items = []
        for case in cases:
            source_id = (
                str(case.get("shuffled_episode_id") or case["episode_id"])
                if arm == "sample_shuffled"
                else str(case["episode_id"])
            )
            source = by_episode.get(source_id, case)
            trajectory = source.get("real_conversation") or []
            if trajectory:
                items.append(
                    {
                        "key": f"{arm}::{case['episode_id']}",
                        "trajectory": trajectory,
                    }
                )
    else:
        items = [
            {
                "key": f"{arm}::{case['episode_id']}",
                "cue_embedding": case.get("cue_embedding"),
            }
            for case in cases
            if case.get("cue_embedding") is not None
        ]
    if not items:
        return {}
    from cue_training.evaluation.common.user_sims import example_retrieval_kwargs

    max_new = int(getattr(args, "command_max_new_tokens", 0) or 1024)
    temp = float(getattr(args, "decode_temperature", 0.0) or 0.0)
    top_p = float(getattr(args, "decode_top_p", 1.0) or 1.0)
    n_cand = int(getattr(args, "decode_num_candidates", 1) or 1)
    mode = str(getattr(args, "decode", "sample") or "sample")
    bsz = int(getattr(args, "decode_batch_size", 16) or 16)
    retrieval = example_retrieval_kwargs(args)
    # Retrieval injects examples into the chosen manual after decode; it no longer forces
    # single-candidate sample (multi-candidate sample/natural still work).

    def _decode_items(
        chunk: list[dict[str, Any]],
        *,
        temperature: float,
        num_candidates: int,
        decode_mode: str,
        batch_size: int,
        retrieval_kwargs: dict[str, Any],
    ) -> dict[str, str]:
        if getattr(args, "sidecar_url", None):
            payload = {
                "decoder_dir": args.decoder_dir,
                "items": chunk,
                "max_new_tokens": max_new,
                "temperature": temperature,
                "top_p": top_p,
                "num_candidates": num_candidates,
                "decode_mode": decode_mode,
                "decode_slot_dedup_jaccard": float(
                    getattr(args, "decode_slot_dedup_jaccard", 0.5) or 0.5
                ),
                "decode_noop_retries": int(getattr(args, "decode_noop_retries", 0) or 0),
                "decode_noop_temperature": float(
                    getattr(args, "decode_noop_temperature", 0.7) or 0.7
                ),
                "session_preprocess": str(
                    getattr(args, "encode_session_preprocess", "full") or "full"
                ),
                "batch_size": batch_size,
                **retrieval_kwargs,
            }
            body = post_sidecar_json(
                f"{str(args.sidecar_url).rstrip('/')}/decode_manuals",
                payload,
                timeout=max(3600.0, 180.0 * max(len(chunk), 1)),
                tag="simarena:run",
            )
            if body.get("error"):
                raise SidecarError(f"CUE pre-decode failed for {arm}: {body['error']}")
            return {
                str(k): str(v)
                for k, v in (body.get("manuals") or {}).items()
            }
        from cue_training.evaluation.common.user_sims import decode_manuals_batch

        return {
            str(k): str(v)
            for k, v in decode_manuals_batch(
                decoder_dir=str(args.decoder_dir),
                items=[
                    (
                        it["key"],
                        it["trajectory"] if live_encode else it["cue_embedding"],
                    )
                    for it in chunk
                ],
                device=getattr(args, "device", "cuda"),
                max_new_tokens=max_new,
                temperature=temperature,
                top_p=top_p,
                num_candidates=num_candidates,
                decode_mode=decode_mode,
                decode_slot_dedup_jaccard=float(
                    getattr(args, "decode_slot_dedup_jaccard", 0.5) or 0.5
                ),
                decode_noop_retries=int(getattr(args, "decode_noop_retries", 0) or 0),
                decode_noop_temperature=float(
                    getattr(args, "decode_noop_temperature", 0.7) or 0.7
                ),
                session_preprocess=str(
                    getattr(args, "encode_session_preprocess", "full") or "full"
                ),
                batch_size=batch_size,
                **retrieval_kwargs,
            ).items()
        }

    out: dict[str, str] = {}
    if getattr(args, "sidecar_url", None):
        chunk_size = max(int(bsz), 32)
        n_chunks = (len(items) + chunk_size - 1) // chunk_size
        for ci in range(n_chunks):
            chunk = items[ci * chunk_size : (ci + 1) * chunk_size]
            out.update(
                _decode_items(
                    chunk,
                    temperature=temp,
                    num_candidates=n_cand,
                    decode_mode=mode,
                    batch_size=bsz,
                    retrieval_kwargs=retrieval,
                )
            )
            log(
                "simarena:run",
                f"pre-decode chunk {ci + 1}/{n_chunks} ({len(out)}/{len(items)} manuals)",
            )
    else:
        out.update(
            _decode_items(
                items,
                temperature=temp,
                num_candidates=n_cand,
                decode_mode=mode,
                batch_size=bsz,
                retrieval_kwargs=retrieval,
            )
        )
        log("simarena:run", f"pre-decoded {len(out)} manuals in-process ({arm})")

    bad = [it["key"] for it in items if not str(out.get(it["key"]) or "").strip()]
    if bad:
        warn(
            "simarena:run",
            f"{arm}: {len(bad)}/{len(items)} manuals empty after greedy pre-decode; "
            f"retrying with natural multi-candidate (e.g. {bad[:3]})",
        )
        by_key = {it["key"]: it for it in items}
        retried = _decode_items(
            [by_key[k] for k in bad],
            temperature=max(temp, 0.7),
            num_candidates=max(n_cand, 4),
            decode_mode="sample",
            batch_size=max(1, min(8, bsz)),
            retrieval_kwargs={
                "example_retrieval_enabled": False,
                "example_retrieval_k_sessions": 8,
                "example_retrieval_n_general": 2,
                "example_retrieval_n_specific": 2,
            },
        )
        for key, manual in retried.items():
            if str(manual).strip():
                out[key] = manual
    still_bad = [it["key"] for it in items if not str(out.get(it["key"]) or "").strip()]
    if still_bad:
        warn(
            "simarena:run",
            f"{arm}: dropping {len(still_bad)}/{len(items)} episodes with empty manuals "
            f"after retry (e.g. {still_bad[:3]})",
        )
        for key in still_bad:
            out.pop(key, None)
    if not out:
        raise SidecarError(
            f"CUE pre-decode returned 0/{len(items)} usable manuals for {arm}"
        )
    log("simarena:run", f"pre-decoded {len(out)}/{len(items)} usable manuals ({arm})")
    return {str(key): str(manual) for key, manual in out.items() if str(manual).strip()}


def run(args: argparse.Namespace) -> None:
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    domain = args.domain
    annotations = load_annotations(args.annotations)
    # Only CUE / retrieval variants need the embedding join; persona/as_is baselines don't.
    from cue_training.evaluation.common.sampler_arms import has_sampled_arms, sampler_kwargs_from_args

    require_embeddings = (
        any(_uses_cue_cases(v) for v in args.variants)
        and not args.live_encode_trajectories
    ) or has_sampled_arms(args.arms)
    cases, counts = build_cases(
        annotations,
        args.embeddings,
        arms=tuple(args.arms),
        background_path=args.background,
        profiles_dir=args.profiles_dir,
        seed=args.seed,
        domain=domain,
        require_embeddings=require_embeddings,
        **sampler_kwargs_from_args(args),
    )
    (out_dir / "prep_counts.json").write_text(json.dumps(counts, indent=2), encoding="utf-8")
    log("simarena:run", f"cases ({domain}): {json.dumps(counts)}")

    persona_map = read_persona_map(args.personas) if args.personas else {}

    match_dataset = str(args.assistant_model).strip().lower() in _DATASET_ASSISTANT_TOKENS
    assistant_router = None
    if match_dataset:
        from cue_training.evaluation.common.dataset_assistants import DatasetAssistantRouter, resolve_simarena_assistant

        assistant_router = DatasetAssistantRouter(
            resolve_simarena_assistant(case.get("assistant_model"))
            for arm_cases in cases.values()
            for case in arm_cases
        )

    commit = None
    try:
        commit = bridge.commit(bridge.repo_path())
    except Exception:  # noqa: BLE001
        pass

    results_path = out_dir / "results.jsonl"
    rollout_path = out_dir / "rollout.simulatorarena.jsonl"
    write_rollouts = not args.no_mirrorbench_rollouts
    done = prepare_resume(out_dir, keep_rollouts=write_rollouts)
    if done:
        log("simarena:run", f"resume: skipping {len(done)} terminal episode(s) already on disk")

    error_log = (
        Path(args.error_log)
        if getattr(args, "error_log", None)
        else out_dir / "logs" / "episode_errors.jsonl"
    )
    error_log.parent.mkdir(parents=True, exist_ok=True)
    if not done:
        error_log.unlink(missing_ok=True)

    write_lock = threading.Lock()

    def _append_result(row: dict[str, Any]) -> None:
        with write_lock:
            append_jsonl(str(results_path), [row])

    def _append_rollout(row: dict[str, Any]) -> None:
        with write_lock:
            append_jsonl(str(rollout_path), [row])

    def _record_error(
        case: dict[str, Any],
        alt: dict[str, Any],
        exc: BaseException,
        variant: str,
        arm: str,
        command_block: str | None = None,
    ) -> None:
        blocked = _is_content_policy_error(exc)
        row = {
            "variant": variant,
            "arm": arm,
            "episode_id": case["episode_id"],
            "status": "error",
            "error": repr(exc),
            "content_policy_blocked": blocked,
        }
        _append_result(row)
        detail = {
            **row,
            "timestamp_utc": dt.datetime.now(dt.UTC).isoformat(),
            "domain": domain,
            "exception_type": type(exc).__name__,
            "traceback": "".join(traceback.format_exception(exc)),
            "case": case,
            "alternate_case": alt,
            "command_block": command_block,
        }
        with write_lock:
            with error_log.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(detail, ensure_ascii=False, default=str) + "\n")

    for variant in args.variants:
        extra_base = _variant_extra(variant, args)
        name = _baseline_name(variant)
        is_persona = variant.startswith("baseline:") and name in PERSONA_BASELINES
        uses_cue = _uses_cue_cases(variant)
        for arm in _arms_for(variant, args.arms):
            # CUE variants pick the per-arm embedding case set; persona / as_is baselines reuse
            # the paired episode set -- the arm only selects the injected profile/command
            # palette, not the embedding.
            case_arm = arm if (
                uses_cue
                and arm in ("paired", "sample_shuffled", "dataset_mean", "sampled_uncond", "sampled_pop")
            ) else "paired"
            # Non-CUE baselines reuse one episode set; fall back to the arm actually built
            # (e.g. as_is for userlm/ppol) when the canonical "paired" set wasn't produced.
            if case_arm not in cases:
                case_arm = arm if arm in cases else next(iter(cases), case_arm)
            arm_cases = cases.get(case_arm, [])
            if args.episode_ids:
                keep = set(args.episode_ids)
                arm_cases = [c for c in arm_cases if c["episode_id"] in keep]
            if args.max_conversations:
                arm_cases = arm_cases[: args.max_conversations]
            by_episode = {str(c["episode_id"]): c for c in arm_cases}
            # CUE decoder: batch-decode this arm's manuals up front (one GPU pass) so per-episode
            # calls skip the serialized decode (matches the tau2 runner).
            manuals = (
                _predecode_manuals(args, arm, arm_cases)
                if variant == "decoder" and getattr(args, "decoder_dir", None)
                else {}
            )
            if manuals:
                before = len(arm_cases)
                arm_cases = [
                    c
                    for c in arm_cases
                    if str(manuals.get(f"{arm}::{c['episode_id']}") or "").strip()
                ]
                by_episode = {str(c["episode_id"]): c for c in arm_cases}
                if len(arm_cases) < before:
                    warn(
                        "simarena:run",
                        f"{variant}/{arm}: running {len(arm_cases)}/{before} cases "
                        "(skipped empty-manual episodes)",
                    )
            # USP Diverse Profile Sampling: sidecar rows win; else GKDE+OF/SC sample once per arm.
            diverse_by_eid: dict[str, str] = {}
            if is_persona and arm == "sample_diverse" and _baseline_name(variant) == "usp":
                need = [
                    c for c in arm_cases
                    if persona_map.get((variant, arm, str(c["episode_id"]))) is None
                ]
                if need:
                    try:
                        from cue_training.baselines.usp.diverse import build_diverse_texts_from_lmsys

                        texts = build_diverse_texts_from_lmsys(len(arm_cases), seed=args.seed)
                        if len(texts) != len(arm_cases):
                            raise ValueError(
                                f"diverse sampler returned {len(texts)} texts for "
                                f"{len(arm_cases)} cases"
                            )
                        diverse_by_eid = {
                            str(c["episode_id"]): texts[i] for i, c in enumerate(arm_cases)
                        }
                    except Exception as exc:  # noqa: BLE001
                        warn("simarena:run", f"sample_diverse build failed: {exc}")

            def _one(case: dict[str, Any], alt: dict[str, Any], _variant: str = variant, _arm: str = arm):
                persona = None
                if is_persona:
                    if _arm == "paired":
                        persona = case.get("profile")
                    elif _arm == "sample_shuffled":
                        persona = alt.get("profile")
                    elif _arm == "sample_diverse":
                        persona = diverse_by_eid.get(str(case["episode_id"]))
                mapped = persona_map.get((_variant, _arm, str(case["episode_id"])))
                if mapped is not None:  # external sidecar (e.g. ppol G(c,D,N) / USP sample_diverse) overrides
                    persona = mapped
                if is_persona and _arm == "sample_diverse" and persona is None:
                    persona = alt.get("profile") or case.get("profile")
                extra = dict(extra_base)
                if args.sidecar_url:
                    extra["sidecar_url"] = args.sidecar_url
                if _variant == "decoder":
                    extra["command_block"] = manuals[f"{_arm}::{case['episode_id']}"]
                # SimulatorArena's native Thought/Message CoT: its template carries the
                # "share materials gradually" guideline and the "terminate conversation"
                # contract, which the baselines prompt lacks (turn-1 info dumps, no stop).
                # Applied to every arm that can accept a prompt override -- CUE and the
                # prompt-based baselines then differ only in persona, not in format, so
                # their metrics are comparable. Trained baselines with a fixed system-prompt
                # contract (UserLM, USP adapter) ignore it and keep their own.
                extra.update({
                    "simarena_native_prompt": True,
                    "document_type": case.get("document_type"),
                    "intent": case.get("intent"),
                    "background": case.get("background"),
                    "math_problem": case.get("math_problem"),
                })
                use_cue = uses_cue and _arm != "sample_random"
                ctx = TurnContext(
                    variant=_variant, arm=_arm, task=case["task_description"], domain=domain,
                    episode_id=case["episode_id"],
                    cue_embedding=case.get("cue_embedding") if use_cue else None,
                    persona=persona, extra=extra,
                )
                sim = build_user_sim(ctx)
                if match_dataset:
                    from cue_training.evaluation.common.dataset_assistants import resolve_simarena_assistant

                    assistant_model = resolve_simarena_assistant(case.get("assistant_model"))
                else:
                    assistant_model = args.assistant_model
                return simulate_case(
                    user_sim=sim, assistant_model=assistant_model,
                    max_turns=args.max_turns, domain=domain, task=case["task_description"],
                    assistant_router=assistant_router,
                )

            def _record(
                case: dict[str, Any],
                alt: dict[str, Any],
                conversation: list[dict[str, str]],
                _variant: str = variant,
                _arm: str = arm,
            ) -> str:
                if not has_nonempty_user_turn(conversation):
                    _append_result({
                        "variant": _variant, "arm": _arm, "episode_id": case["episode_id"],
                        "assistant_model": case["assistant_model"],
                        "document_type": case.get("document_type"),
                        "intent": case.get("intent"), "problem_id": case.get("problem_id"),
                        "num_turns": len(conversation), "status": "error",
                        "error": "empty_user_sim_response",
                    })
                    return "error"
                _append_result({
                    "variant": _variant, "arm": _arm, "episode_id": case["episode_id"],
                    "assistant_model": case["assistant_model"], "document_type": case.get("document_type"),
                    "intent": case.get("intent"), "problem_id": case.get("problem_id"),
                    "num_turns": len(conversation), "status": "ok",
                })
                if write_rollouts and conversation:
                    if _arm == "paired":
                        origin = case["episode_id"]
                    elif _arm == "sample_shuffled":
                        origin = alt.get("episode_id")
                    else:
                        origin = None
                    _append_rollout(to_mirrorbench_rollout(
                        source_record={
                            "episode_id": case["episode_id"], "task_id": case["episode_id"],
                            "dataset": "simulatorarena", "real_conversation": case.get("real_conversation") or [],
                            "task_description": case["task_description"], "metadata": {"domain": domain},
                        },
                        conversation=conversation,
                        provenance={
                            "source_harness": "simulatorarena", "variant": _variant, "arm": _arm,
                            "assistant_model": case["assistant_model"], "document_type": case.get("document_type"),
                            "intent": case.get("intent"), "problem_id": case.get("problem_id"),
                            "workerId": case.get("workerId"),
                            "annotation_key": case["annotation_key"], "simarena_commit": commit,
                            "persona_origin_episode_id": origin,
                            "shuffled_episode_id": alt.get("episode_id") if _arm == "sample_shuffled" else None,
                        },
                    ))
                return "ok"

            workers = max(1, int(args.concurrency))
            pairs: list[tuple[dict[str, Any], dict[str, Any]]] = []
            for case in arm_cases:
                if arm == "sample_shuffled":
                    origin_id = str(case.get("shuffled_episode_id") or case["episode_id"])
                    alt = by_episode.get(origin_id) or case
                else:
                    alt = case
                pairs.append((case, alt))
            n_prior = sum(
                1 for case, _alt in pairs
                if _episode_key(variant, arm, case["episode_id"]) in done
            )
            pairs = [
                (case, alt) for case, alt in pairs
                if _episode_key(variant, arm, case["episode_id"]) not in done
            ]
            log(
                "simarena:run",
                f"{variant}/{arm}: {len(pairs)} conversations to run "
                f"(skipped {n_prior} resumed), concurrency={workers}",
            )
            if not pairs:
                continue
            n_ok = 0
            n_err = 0
            bar = progress(
                total=len(pairs), desc=f"simarena:{variant}/{arm}", unit="ep",
                tag="simarena:run", leave=True,
            )

            def _finish(case: dict[str, Any], alt: dict[str, Any], outcome: Any) -> None:
                nonlocal n_ok, n_err
                try:
                    status = _record(case, alt, outcome() if callable(outcome) else outcome)
                    if status == "ok":
                        n_ok += 1
                        done.add(_episode_key(variant, arm, case["episode_id"]))
                    else:
                        n_err += 1
                except SidecarError as exc:
                    # A broken decoder/sidecar stays fail-closed. Content-policy blocks and
                    # provider blips (disconnect / 5xx) are logged per-episode and skipped
                    # instead, so one flaky upstream call cannot discard the whole run.
                    if not (_is_content_policy_error(exc) or is_transient_provider_error(exc)):
                        raise
                    if not _is_content_policy_error(exc):
                        warn("simarena:run", f"transient provider error; skipping episode: {exc}")
                    _record_error(
                        case, alt, exc, variant, arm,
                        manuals.get(f"{arm}::{case['episode_id']}"),
                    )
                    n_err += 1
                    if _is_content_policy_error(exc):
                        done.add(_episode_key(variant, arm, case["episode_id"]))
                except Exception as exc:  # noqa: BLE001
                    _record_error(
                        case, alt, exc, variant, arm,
                        manuals.get(f"{arm}::{case['episode_id']}"),
                    )
                    n_err += 1
                finally:
                    bar.update(1)
                    bar.set_postfix(
                        ok=n_ok,
                        err=n_err,
                        err_pct=f"{100.0 * n_err / max(1, n_ok + n_err):.2f}%",
                    )

            try:
                if workers == 1:
                    for case, alt in pairs:
                        _finish(case, alt, lambda c=case, a=alt: _one(c, a))
                else:
                    # Merge results on the main thread (append paths take the write lock).
                    with ThreadPoolExecutor(max_workers=workers) as pool:
                        futs = {pool.submit(_one, case, alt): (case, alt) for case, alt in pairs}
                        for fut in as_completed(futs):
                            case, alt = futs[fut]
                            _finish(case, alt, fut.result)
            finally:
                bar.close()

    results = read_jsonl(str(results_path)) if results_path.is_file() else []
    (out_dir / "summary.json").write_text(json.dumps(_summarize(results), indent=2), encoding="utf-8")
    if write_rollouts:
        _rebuild_native_from_rollouts(out_dir, domain)
    log("simarena:run", f"wrote {len(results)} results -> {results_path}")


def _accumulate_native(
    native: dict[str, dict[str, Any]], variant: str, arm: str, case: dict, conversation: list[dict],
    domain: str = "writing",
) -> None:
    """Accumulate SimulatorArena-format output. Writing nests [model][DocType][intent][workerId];
    math nests [model][problem_id][workerId].

    NOTE: ``conversation`` is emitted as a list of ``[role, text]`` pairs to mirror
    SimulatorArena's tuple format; reconcile any remaining fields its terminate/eval
    scripts expect during execution against the submodule.
    """

    bucket = native.setdefault(f"{variant}__{arm}", {})
    model = str(case["assistant_model"])
    worker = str(case["workerId"])
    record = {
        "conversation": [[t["role"], t["content"]] for t in conversation],
        "user_queries": [t["content"] for t in conversation if t["role"] == "user"],
        "ai_responses": [t["content"] for t in conversation if t["role"] == "assistant"],
    }
    if domain == "math":
        problem = str(case.get("problem_id"))
        bucket.setdefault(model, {}).setdefault(problem, {})[worker] = record
        return
    doc_type = _DOC_TYPE_MAP.get(str(case.get("document_type")).lower(), str(case.get("document_type")))
    intent = str(case.get("intent"))
    bucket.setdefault(model, {}).setdefault(doc_type, {}).setdefault(intent, {})[worker] = record


def read_persona_map(path: str) -> dict[tuple[str, str, str], Any]:
    """Optional persona sidecar: rows of {variant, arm, episode_id, persona}."""

    from cue_training.evaluation.common.io import read_jsonl

    out: dict[tuple[str, str, str], Any] = {}
    for row in read_jsonl(path):
        out[(row["variant"], row["arm"], str(row["episode_id"]))] = row.get("persona")
    return out


def _write_native(out_dir: Path, native: dict[str, dict[str, Any]]) -> None:
    native_dir = out_dir / "native"
    native_dir.mkdir(parents=True, exist_ok=True)
    for label, output in native.items():
        (native_dir / f"{label}.json").write_text(json.dumps(output, indent=2), encoding="utf-8")


def _summarize(results: list[dict[str, Any]]) -> dict[str, Any]:
    from collections import defaultdict

    ok = [r for r in results if r.get("status") == "ok"]
    by = defaultdict(list)
    for r in ok:
        by[(r["variant"], r["arm"])].append(r)
    return {
        "n_results": len(results), "n_ok": len(ok),
        "n_error": sum(1 for r in results if r.get("status") == "error"),
        "avg_turns": {f"{v}|{a}": sum(x["num_turns"] for x in items) / len(items)
                      for (v, a), items in by.items()},
    }


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="SimulatorArena eval (writing/math) with CUE/baseline user sims.")
    p.add_argument("--annotations", required=True,
                   help="document_creation_annotations[_for_benchmarking].json (writing) or "
                        "math_tutoring_annotations[_for_benchmarking].json (math).")
    p.add_argument("--embeddings", required=True, help="CUE encoder export over the simarena domain")
    p.add_argument(
        "--live_encode_trajectories",
        action="store_true",
        help="Encode each reference trajectory with the loaded joint training checkpoint.",
    )
    p.add_argument(
        "--encode_session_preprocess",
        default="full",
        choices=("full", "strip_document", "user_only"),
        help="Preprocess reference turns before live-encode (default: full).",
    )
    p.add_argument("--out_dir", required=True)
    p.add_argument(
        "--error_log",
        default=None,
        help="JSONL path for complete per-episode errors (default: <out_dir>/logs/episode_errors.jsonl).",
    )
    p.add_argument("--domain", choices=["writing", "math"], default="writing",
                   help="SimulatorArena task domain.")
    p.add_argument("--background", default=None, help="document_creation_user_simulator_background.json (writing WHAT)")
    p.add_argument("--profiles_dir", default=None,
                   help="user_simulator_profiles/<document_creation|math_tutoring> dir (paired/shuffled persona)")
    p.add_argument("--variants", nargs="+", default=["decoder", "baseline:realusersim", "base_api"])
    p.add_argument("--arms", nargs="+", default=["paired", "sample_shuffled", "dataset_mean"])
    p.add_argument("--assistant_model", default="gpt-5.2",
                   help="Assistant model that actually drives replies (litellm id). The arena's "
                        "original per-annotation model is kept only as metadata/grouping.")
    p.add_argument("--sim_model", default="gpt-5.4-mini")
    p.add_argument("--sim_api_base", default=None,
                   help="OpenAI-compatible base URL for the decoder/baseline simulator (e.g. a vLLM "
                        "endpoint). Default: the provider API.")
    p.add_argument("--sim_api_key_env", default="OPENAI_API_KEY",
                   help="Env var holding the API key used with --sim_api_base.")
    p.add_argument("--sim_max_tokens", type=int, default=512,
                   help="Max completion tokens for the user simulator (from simulators[].sampling).")
    p.add_argument("--sim_temperature", type=float, default=0.7,
                   help="Sampling temperature for the user simulator.")
    p.add_argument("--sim_top_p", type=float, default=0.95,
                   help="Nucleus sampling top_p for the user simulator.")
    p.add_argument("--semantic_model", default="intfloat/e5-base-v2")
    p.add_argument("--decoder_dir", default=None)
    p.add_argument("--decode_temperature", type=float, default=0.0,
                   help="Command-block decode temperature for the decoder (0.0=greedy).")
    p.add_argument("--decode_top_p", type=float, default=1.0,
                   help="Nucleus sampling probability for decoder generation.")
    p.add_argument("--decode_num_candidates", type=int, default=1,
                   help="Manuals to sample per persona; >1 reranks for naturalness (samples even "
                   "when decode_temperature=0).")
    p.add_argument("--decode", choices=("greedy", "sample", "diverse_slots"),
                   default="sample",
                   help="Manual selection: greedy/sample, or diverse_slots "
                        "(per-slot threshold dedup; decode_num_candidates = samples per slot).")
    p.add_argument("--decode_slot_dedup_jaccard", type=float, default=0.5,
                   help="For --decode diverse_slots: reject candidates with token-Jaccard ≥ this "
                        "vs an already-kept command in the same head.")
    p.add_argument("--decode_noop_retries", type=int, default=0,
                   help="Resample slots that greedily decode to <NO_COMMAND> this many times "
                        "instead of leaving the slot empty.")
    p.add_argument("--decode_noop_temperature", type=float, default=0.7,
                   help="Sampling temperature for --decode_noop_retries.")
    p.add_argument("--decode_batch_size", type=int, default=16,
                   help="Total sequences per GPU pass (personas x candidates) during manual "
                   "pre-decode; raise to use more VRAM and speed up decoding.")
    p.add_argument(
        "--example_retrieval_enabled",
        action="store_true",
        help="Inject nearest-neighbor style examples from the joint training example_pool.",
    )
    p.add_argument("--example_retrieval_k_sessions", type=int, default=8)
    p.add_argument("--example_retrieval_n_general", type=int, default=2)
    p.add_argument("--example_retrieval_n_specific", type=int, default=2)
    p.add_argument("--baseline_output_dir", default=str(storage_root() / "baselines"))
    p.add_argument("--baseline_artifacts_dir", default=None)
    p.add_argument("--userlm_vllm_base_url", default=None,
                   help="If set, baseline:userlm calls this vLLM endpoint in-process (no sidecar).")
    p.add_argument("--userlm_vllm_model", default="userlm")
    p.add_argument("--userlm_vllm_api_key_env", default="HOSTED_VLLM_API_KEY")
    p.add_argument("--usp_vllm_base_url", default=None,
                   help="If set, baseline:usp calls this vLLM endpoint in-process (no sidecar).")
    p.add_argument("--usp_vllm_model", default="wangkevin02/USP")
    p.add_argument("--usp_vllm_api_key_env", default="HOSTED_VLLM_API_KEY")
    p.add_argument("--base_api_model", default="gpt-5.4-mini")
    p.add_argument("--base_local_path", default="meta-llama/Llama-3.1-8B-Instruct")
    p.add_argument("--vllm_base_url", default=None,
                   help="If set, base_local calls this vLLM OpenAI endpoint in-process (no sidecar).")
    p.add_argument("--sidecar_url", default=None,
                   help="If set, GPU variants generate via the shared CUE sidecar (CPU-only head).")
    p.add_argument("--personas", default=None,
                   help="Optional persona sidecar keyed by "
                        "(variant, arm=paired|sample_shuffled|sample_diverse, episode_id).")
    p.add_argument("--episode_ids", nargs="*", default=None, help="Episode-id subset (e.g. sharded runs).")
    p.add_argument("--max_conversations", type=int, default=None)
    p.add_argument("--max_turns", type=int, default=12)
    p.add_argument("--concurrency", type=int, default=8,
                   help="Episode-level fan-out (threads). vLLM/API sims batch server-side.")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default="cuda")
    p.add_argument("--dtype", default="bfloat16")
    p.add_argument("--no_mirrorbench_rollouts", action="store_true")
    from cue_training.evaluation.common.sampler_arms import add_sampler_argparse

    add_sampler_argparse(p)
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    run(parse_args(argv))
