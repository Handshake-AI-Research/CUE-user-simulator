"""Run tau2 tasks with our user simulators and collect task-success + MirrorBench rollouts.

For each (variant, arm, episode): build the per-task conditioning, run tau2 closed-loop
with the registered ``cue_eval_user``, record reward/success, and (by default) also emit
a tool-stripped MirrorBench rollout record for reuse by the MirrorBench pipeline.
"""

from __future__ import annotations

import argparse
import json
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

from cue_training.utils.config import storage_root
from cue_training.evaluation.common.io import read_jsonl
from cue_training.evaluation.rollouts.artifacts import atomic_write_jsonl, promote_final_artifacts
from cue_training.evaluation.common.mirrorbench_export import has_nonempty_user_turn, to_mirrorbench_rollout
from cue_training.evaluation.common.user_sims import SidecarError, reset_sidecar_errors, sidecar_errors
from cue_training.evaluation.tau2_bench import bridge
from cue_training.evaluation.tau2_bench.data import scenario_only
from cue_training.runlog.log import log, progress

# variant -> arms it runs, and whether it uses the CUE embedding.
CUE_VARIANTS = {"decoder"}
PERSONA_BASELINES = {"usp", "realusersim_paired_noex"}  # paper realusersim is as_is WildChat-pool sampling
PAIRED_REALUSERSIM = {"realusersim_paired_noex"}
AS_IS_BASELINES = {"ppol", "userlm"}


def _baseline_name(variant: str) -> str:
    return variant.split(":", 1)[1] if variant.startswith("baseline:") else variant


def _uses_cue_cases(variant: str) -> bool:
    return variant in CUE_VARIANTS


def _sim_sampling_extra(args: argparse.Namespace) -> dict[str, Any]:
    """Forward simulator generation knobs into user-sim ``extra`` (defaults match prior hardcoded)."""

    return {
        "sim_max_tokens": int(getattr(args, "sim_max_tokens", 512)),
        "sim_temperature": float(getattr(args, "sim_temperature", 0.7)),
        "sim_top_p": float(getattr(args, "sim_top_p", 0.95)),
    }


def _variant_extra(variant: str, args: argparse.Namespace) -> dict[str, Any]:
    """Model paths + generation knobs handed to the user sim via llm_args.extra."""

    common = {"device": args.device, "dtype": args.dtype, **_sim_sampling_extra(args)}
    if variant == "decoder":
        from cue_training.evaluation.common.user_sims import example_retrieval_kwargs

        return {**common, "decoder_dir": args.decoder_dir, "semantic_model": args.semantic_model,
                "sim_model": args.sim_model, "sim_api_base": args.sim_api_base,
                "sim_api_key_env": args.sim_api_key_env,
                "decode_temperature": args.decode_temperature,
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
        # Policy vLLM: workers call the OpenAI server in-process (no sidecar).
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


def _arms_for(variant: str, requested_arms: list[str]) -> list[str]:
    if variant in CUE_VARIANTS:
        return [
            a for a in requested_arms
            if a in ("paired", "sample_shuffled", "dataset_mean", "sampled_uncond", "sampled_pop")
        ] or ["paired"]
    name = _baseline_name(variant)
    if name in PERSONA_BASELINES:
        if name in PAIRED_REALUSERSIM:
            allowed = ("paired", "sample_shuffled")
            return [a for a in requested_arms if a in allowed] or ["paired"]
        # USP defaults: paired + sample_diverse (sample_shuffled still allowed if requested).
        allowed = ("paired", "sample_shuffled", "sample_diverse")
        return [a for a in requested_arms if a in allowed] or ["paired", "sample_diverse"]
    return ["as_is"]


def _cases_for_arm(cases_dir: Path, arm: str, variant: str) -> list[dict[str, Any]]:
    # CUE decoders use per-arm case files (different cue_embedding).
    # Persona baselines reuse the canonical paired episode set; base/as_is may write
    # cases.as_is.jsonl (no embeddings) or cases.paired.jsonl.
    use_cue = _uses_cue_cases(variant)
    if use_cue and arm == "sample_random":
        # Random profile ignores the cue; still need a stable episode list.
        arm = "paired"
    if use_cue and arm in ("paired", "sample_shuffled", "dataset_mean", "sampled_uncond", "sampled_pop"):
        candidates = [arm]
    elif arm == "as_is":
        candidates = ["as_is", "paired"]
    else:
        candidates = ["paired", "as_is"]
    for fname in candidates:
        path = cases_dir / f"cases.{fname}.jsonl"
        if path.exists():
            rows = read_jsonl(str(path))
            if rows:
                return rows
    return []


def _predecode_manuals(
    args: argparse.Namespace, arm: str, cases: list[dict[str, Any]]
) -> dict[str, str]:
    """Batch-decode this arm's persona manuals up front (on the sidecar if used, else in-process),
    caching them so per-episode next_turn calls skip the GPU decode."""

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
    # Retrieval injects examples after the chosen manual is selected; keep caller's
    # decode_mode / decode_num_candidates (including sample with N>1).
    if args.sidecar_url:
        from cue_training.evaluation.common.user_sims import post_sidecar_json

        # Chunked like prism: one POST for a whole arm outgrows the read timeout.
        # Keep chunks small enough that even dual-decode + retrieval finishes well under the
        # per-request timeout; scale timeout with chunk size.
        chunk_size = max(int(bsz), 32)
        n_chunks = (len(items) + chunk_size - 1) // chunk_size
        out: dict[str, Any] = {}
        for ci in range(n_chunks):
            chunk = items[ci * chunk_size : (ci + 1) * chunk_size]
            payload = {
                "decoder_dir": args.decoder_dir,
                "items": chunk,
                "max_new_tokens": max_new,
                "temperature": temp,
                "top_p": top_p,
                "num_candidates": n_cand,
                "decode_mode": mode,
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
                "batch_size": bsz,
                **retrieval,
            }
            # Dual decode (~3 heads x 1024) at batch_size=16 is ~2–4 min/chunk; leave headroom.
            body = post_sidecar_json(
                f"{str(args.sidecar_url).rstrip('/')}/decode_manuals",
                payload,
                timeout=max(3600.0, 180.0 * len(chunk)),
                tag="tau2:run",
            )
            if body.get("error"):
                raise SidecarError(
                    f"CUE pre-decode failed for {arm} chunk {ci + 1}/{n_chunks}: {body['error']}"
                )
            out.update(body.get("manuals") or {})
            log("tau2:run", f"pre-decode chunk {ci + 1}/{n_chunks} ({len(out)}/{len(items)} manuals)")
        empty = [key for key, manual in out.items() if not str(manual).strip()]
        if len(out) != len(items) or empty:
            raise SidecarError(
                f"CUE pre-decode returned {len(out) - len(empty)}/{len(items)} usable manuals "
                f"for {arm}; empty={empty[:3]}"
            )
        log("tau2:run", f"pre-decoded {len(out)} manuals on sidecar ({arm})")
    else:
        from cue_training.evaluation.common.user_sims import decode_manuals_batch

        out = decode_manuals_batch(
            decoder_dir=str(args.decoder_dir),
            items=[
                (
                    it["key"],
                    it["trajectory"] if live_encode else it["cue_embedding"],
                )
                for it in items
            ],
            device=args.device, dtype=args.dtype, max_new_tokens=max_new, temperature=temp,
            top_p=top_p,
            num_candidates=n_cand, decode_mode=mode, batch_size=bsz,
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
            **retrieval,
        )
        log("tau2:run", f"pre-decoded {len(out)} manuals in-process ({arm})")
    return {str(key): str(manual) for key, manual in out.items()}


def _run_one_case(
    *,
    settings: bridge.Tau2Settings,
    case: dict[str, Any],
    variant: str,
    arm: str,
    extra_base: dict[str, Any],
    personas: dict[tuple[str, str, str], Any],
    args: argparse.Namespace,
    commit: str | None,
    manuals: dict[str, str],
    shuffle_map: dict[str, str] | None = None,
) -> tuple[dict[str, Any], dict[str, Any] | None]:
    """Run one (variant, arm, case); return (result_row, optional mirrorbench rollout)."""

    case = _ensure_shuffle_ids(case, shuffle_map=shuffle_map or {})
    extra = dict(extra_base)
    if args.sidecar_url:
        extra["sidecar_url"] = args.sidecar_url
        if args.sidecar_timeout:
            extra["sidecar_timeout"] = args.sidecar_timeout
    if variant == "decoder":
        extra["command_block"] = manuals[f"{arm}::{case['episode_id']}"]
    scenario = case.get("scenario") or scenario_only(case.get("task_description") or "")
    if scenario:
        extra["task_scenario"] = scenario
    persona = personas.get((variant, arm, case["episode_id"]))
    use_cue = _uses_cue_cases(variant) and arm != "sample_random"
    llm_args = {
        "variant": variant, "arm": arm, "domain": case["tau2_domain"],
        "episode_id": case["episode_id"],
        "cue_embedding": case.get("cue_embedding") if use_cue else None,
        "persona": persona, "extra": extra,
    }
    label = f"{variant}__{arm}__{case['episode_id']}".replace("/", "_")
    try:
        res = bridge.run_single_task(
            settings, domain=case["tau2_domain"], task_id=case["tau2_task_id"],
            instructions=case["task_description"], llm_args_user=llm_args,
            num_trials=args.trials, run_label=label,
        )
    except SidecarError:
        raise  # CUE decoder/sidecar failure: abort the run, don't record a silent error row
    except Exception as exc:  # noqa: BLE001
        return (
            {"variant": variant, "arm": arm, "episode_id": case["episode_id"],
             "status": "error", "error": repr(exc)},
            None,
        )
    row = {
        "variant": variant, "arm": arm, "episode_id": case["episode_id"],
        "tau2_domain": res.domain, "tau2_task_id": res.task_id,
        "found_task": res.found_task, "reward": res.reward, "is_success": res.is_success,
        "termination_reason": res.termination_reason, "num_turns": res.num_turns,
        "save_path": res.save_path, "tau2_commit": commit,
        "reward_report": res.reward_report,
        "status": "ok" if res.found_task and res.reward is not None else "skipped",
        "error": res.error,
    }
    # User-sim failures return ("", done=True); tau2 still emits a "successful" empty
    # dialogue. Treat those as errors so resume re-runs them instead of scoring blanks.
    if not has_nonempty_user_turn(res.conversation):
        row["status"] = "error"
        row["error"] = row.get("error") or "empty_user_sim_response"
        return row, None
    rollout = None
    if not args.no_mirrorbench_rollouts and res.conversation:
        rollout = _mirrorbench_row(
            case, res, variant, arm, commit, shuffle_map=shuffle_map
        )
    return row, rollout


def run(args: argparse.Namespace) -> None:
    cases_dir = Path(args.cases_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    repo = bridge.ensure_tau2_importable(Path(args.tau2_path) if args.tau2_path else None)
    commit = bridge.tau2_commit(repo)
    from cue_training.evaluation.tau2_bench.registry import register_cue_eval_user

    user_name = register_cue_eval_user()
    # tau2 swallows user-sim exceptions, so track CUE sidecar failures out-of-band and abort below.
    reset_sidecar_errors()

    # Episode fan-out is owned by this runner; each tau2 call is a single task (concurrency=1).
    settings = bridge.Tau2Settings(
        repo_path=repo, output_dir=out_dir / "tau2_runs", user=user_name,
        agent_llm=args.agent_llm, max_concurrency=1,
    )
    personas = read_persona_map(args.personas) if args.personas else {}
    from cue_training.evaluation.common.shuffle import build_shuffle_map, domain_of

    # Fallback map for older cases.jsonl that omit shuffled_episode_id.
    all_case_rows: list[dict[str, Any]] = []
    for path in cases_dir.glob("cases.*.jsonl"):
        all_case_rows.extend(read_jsonl(str(path)))
    fallback_ids = []
    fallback_domains: dict[str, str] = {}
    seen_ids: set[str] = set()
    for row in all_case_rows:
        eid = str(row.get("episode_id") or "")
        if not eid or eid in seen_ids:
            continue
        seen_ids.add(eid)
        fallback_ids.append(eid)
        fallback_domains[eid] = domain_of(
            eid, explicit=row.get("tau2_domain") or row.get("domain"), record=row
        )
    fallback_shuffle = build_shuffle_map(
        fallback_ids, domains=fallback_domains, seed=int(getattr(args, "seed", 0) or 0)
    )
    workers = max(1, int(args.concurrency))

    results_path = out_dir / "results.jsonl"
    rollout_rows: list[dict[str, Any]] = []
    results: list[dict[str, Any]] = []

    for variant in args.variants:
        extra_base = _variant_extra(variant, args)
        planned: list[tuple[str, list[dict[str, Any]]]] = []
        for arm in _arms_for(variant, args.arms):
            cases = _cases_for_arm(cases_dir, arm, variant)
            if args.episode_ids:
                keep = set(args.episode_ids)
                cases = [c for c in cases if c["episode_id"] in keep]
            if args.max_episodes:
                cases = cases[: args.max_episodes]
            if not cases:
                continue
            planned.append((arm, cases))

        for arm, cases in planned:
            manuals: dict[str, str] = {}
            if variant == "decoder" and args.decoder_dir:
                log("tau2:run", f"pre-decode {variant}/{arm}: {len(cases)} manuals")
                manuals = _predecode_manuals(args, arm, cases)
            log("tau2:run", f"{variant}/{arm}: {len(cases)} episodes, concurrency={workers}")

            def _job(
                case: dict[str, Any],
                _variant: str = variant,
                _arm: str = arm,
                _manuals: dict[str, str] = manuals,
            ) -> tuple[dict[str, Any], dict[str, Any] | None]:
                return _run_one_case(
                    settings=settings, case=case, variant=_variant, arm=_arm,
                    extra_base=extra_base,
                    personas=personas, args=args, commit=commit, manuals=_manuals,
                    shuffle_map=fallback_shuffle,
                )

            outcomes: list[tuple[dict[str, Any], dict[str, Any] | None]] = []
            n_ok = 0
            n_err = 0
            bar = progress(
                total=len(cases),
                desc=f"tau2:{variant}/{arm}",
                unit="ep",
                tag="tau2:run",
                leave=True,
            )
            try:
                if workers == 1:
                    for case in cases:
                        row, rollout = _job(case)
                        outcomes.append((row, rollout))
                        if row.get("error"):
                            n_err += 1
                        else:
                            n_ok += 1
                        bar.update(1)
                        bar.set_postfix(ok=n_ok, err=n_err)
                else:
                    with ThreadPoolExecutor(max_workers=workers) as pool:
                        futures = [pool.submit(_job, case) for case in cases]
                        for fut in as_completed(futures):
                            row, rollout = fut.result()
                            outcomes.append((row, rollout))
                            if row.get("error"):
                                n_err += 1
                            else:
                                n_ok += 1
                            bar.update(1)
                            bar.set_postfix(ok=n_ok, err=n_err)
            finally:
                bar.close()
            for row, rollout in outcomes:
                results.append(row)
                if rollout is not None:
                    rollout_rows.append(rollout)
            # CUE decoder: a broken sidecar returns silent STOPs (tau2 swallows the exception),
            # so abort BEFORE writing any rollout instead of emitting valid-looking results.
            errs = sidecar_errors()
            if errs:
                raise SidecarError(
                    f"CUE sidecar failed on {len(errs)} call(s); aborting {variant}/{arm}. "
                    f"First: {errs[0]}"
                )
            atomic_write_jsonl(results_path, results)
            _write_summary(out_dir, results)
            if not args.no_mirrorbench_rollouts:
                atomic_write_jsonl(out_dir / "rollout.tau2.jsonl", rollout_rows)
            if args.publish_dir:
                promote_final_artifacts(
                    out_dir,
                    Path(args.publish_dir),
                    ["results.jsonl", "rollout.tau2.jsonl"],
                    replace_arms=[arm] if args.replace_arms else None,
                )
            log("tau2:run", f"checkpointed {variant}/{arm}: {len(results)} results")

    log("tau2:run", f"wrote {len(results)} results -> {results_path}")


def _persona_origin_episode_id(
    case: dict[str, Any],
    arm: str,
    *,
    shuffle_map: dict[str, str] | None = None,
) -> str | None:
    """Episode whose persona conditioned this rollout (None for unconditioned/sampled)."""

    if arm == "paired":
        return str(case["episode_id"])
    if arm == "sample_shuffled":
        sid = case.get("shuffled_episode_id")
        if sid:
            return str(sid)
        if shuffle_map:
            return shuffle_map.get(str(case["episode_id"]))
        return None
    # sample_diverse / dataset_mean / as_is: no single origin episode
    return None


def _ensure_shuffle_ids(
    case: dict[str, Any], *, shuffle_map: dict[str, str]
) -> dict[str, Any]:
    """Copy case with shuffled_episode_id filled from the universal map when missing."""

    if case.get("shuffled_episode_id"):
        return case
    eid = str(case.get("episode_id") or "")
    partner = shuffle_map.get(eid)
    if not partner:
        return case
    return {**case, "shuffled_episode_id": partner}


def _mirrorbench_row(
    case,
    res,
    variant,
    arm,
    commit,
    *,
    shuffle_map: dict[str, str] | None = None,
) -> dict[str, Any]:  # noqa: ANN001
    source = {
        "episode_id": case["episode_id"], "task_id": case["tau2_task_id"],
        "dataset": case["tau2_domain"], "real_conversation": case.get("real_conversation") or [],
        "task_description": case.get("task_description") or "", "intent": case.get("intent") or "",
        "scenario": case.get("scenario") or "", "metadata": {"domain": case["tau2_domain"]},
    }
    origin = _persona_origin_episode_id(case, arm, shuffle_map=shuffle_map)
    shuffled_id = case.get("shuffled_episode_id")
    if not shuffled_id and shuffle_map:
        shuffled_id = shuffle_map.get(str(case["episode_id"]))
    return to_mirrorbench_rollout(
        source_record=source, conversation=res.conversation,
        provenance={
            "source_harness": "tau2", "native_save_path": res.save_path, "variant": variant,
            "arm": arm, "tau2_domain": res.domain, "tau2_task_id": res.task_id,
            "reward": res.reward, "is_success": res.is_success, "tau2_commit": commit,
            "reward_report": res.reward_report,
            # Metrics use stripped dialogue; FMA prefers this tool-aware copy when present.
            "full_conversation": res.conversation,
            "persona_origin_episode_id": origin,
            "shuffled_episode_id": shuffled_id,
        },
    )


def read_persona_map(path: str) -> dict[tuple[str, str, str], Any]:
    """Optional personas sidecar: rows of {variant, arm, episode_id, persona}."""

    out: dict[tuple[str, str, str], Any] = {}
    for row in read_jsonl(path):
        out[(row["variant"], row["arm"], str(row["episode_id"]))] = row.get("persona")
    return out


def _write_summary(out_dir: Path, results: list[dict[str, Any]]) -> None:
    from collections import defaultdict

    scored = [r for r in results if r.get("status") == "ok" and r.get("reward") is not None]
    by = defaultdict(list)
    for r in scored:
        by[(r["variant"], r["arm"], r["tau2_domain"])].append(r)
    summary = {
        "n_results": len(results),
        "n_scored": len(scored),
        "n_error": sum(1 for r in results if r.get("status") == "error"),
        "success_rate": {
            f"{v}|{a}|{d}": sum(1 for r in items if r.get("is_success")) / len(items)
            for (v, a, d), items in by.items()
        },
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run tau2 eval with CUE/baseline user sims.")
    p.add_argument("--cases_dir", required=True)
    p.add_argument("--out_dir", required=True)
    p.add_argument("--publish_dir", default=None)
    p.add_argument("--replace_arms", action="store_true")
    p.add_argument("--tau2_path", default=None)
    p.add_argument("--variants", nargs="+", default=["decoder", "baseline:realusersim", "base_api"])
    p.add_argument("--arms", nargs="+", default=["paired", "sample_shuffled", "dataset_mean"])
    p.add_argument("--agent_llm", default="gpt-5.2")
    p.add_argument("--sim_model", default="gpt-5.4-mini")
    p.add_argument("--sim_api_base", default=None,
                   help="OpenAI-compatible base URL for the decoder simulator (e.g. a vLLM "
                        "endpoint serving an open model). Default: the provider API.")
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
                   help="Command-block decode temperature for the decoder (0.0=greedy). >0 "
                   "samples, restoring per-cue variation if greedy collapses to one block.")
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
    p.add_argument(
        "--live_encode_trajectories",
        action="store_true",
        help="Encode reference trajectories with the loaded joint training checkpoint.",
    )
    p.add_argument(
        "--encode_session_preprocess",
        default="full",
        choices=("full", "strip_document", "user_only"),
        help="Preprocess reference turns before live-encode (default: full).",
    )
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
    p.add_argument("--personas", default=None,
                   help="Optional personas sidecar keyed by "
                        "(variant, arm=paired|sample_shuffled|sample_diverse, episode_id).")
    p.add_argument("--sidecar_url", default=None,
                   help="If set, GPU variants generate via the shared CUE sidecar (CPU-only head).")
    p.add_argument("--sidecar_timeout", type=float, default=None,
                   help="Per-call timeout (s) for sidecar /next_turn requests (default 600). Raise "
                        "if single turns are legitimately slow under heavy fan-out.")
    p.add_argument("--episode_ids", nargs="*", default=None, help="Episode-id subset (e.g. sharded runs).")
    p.add_argument("--trials", type=int, default=1)
    p.add_argument("--max_episodes", type=int, default=None)
    p.add_argument("--concurrency", type=int, default=1)
    p.add_argument("--device", default="cuda")
    p.add_argument("--dtype", default="bfloat16")
    p.add_argument("--no_mirrorbench_rollouts", action="store_true")
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    run(parse_args(argv))
