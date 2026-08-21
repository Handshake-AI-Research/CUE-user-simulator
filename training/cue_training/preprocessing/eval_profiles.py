"""Build data annotation persona sidecars for Tau2, SimulatorArena, or PRISM evaluation cases."""

from __future__ import annotations

import argparse
import json
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

from cue_training.data.schema import (
    canonicalize_profile,
    parse_manual,
    split_manual_targets,
    to_profile,
    validate_manual,
)
from cue_training.preprocessing.data_annotation import build_manual_for_session
from cue_training.preprocessing.dispatch import ReplicaRouter, build_default_router
from cue_training.preprocessing.style_augment import extract_style_commands
from cue_training.refinement.simulator import render_dual_manual
from cue_training.utils.config import get, load_config
from cue_training.utils.log import log, warn

TAG = "eval_profiles"
# Runner keys personas as (variant, arm, episode_id); must match baseline:realusersim.
REALUSERSIM_VARIANT = "baseline:realusersim"


def load_tau2_cases(path: str | Path) -> list[dict[str, Any]]:
    path = Path(path)
    cases = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            cases.append(json.loads(line))
    return cases


def case_to_record(case: dict[str, Any]) -> dict[str, Any]:
    """Convert an eval case with real_conversation into a cue record."""

    conv = case.get("real_conversation") or case.get("conversation") or []
    turns = []
    for i, msg in enumerate(conv):
        role = str(msg.get("role") or "").lower()
        content = str(msg.get("content") or "").strip()
        if role not in {"user", "assistant", "system"} or not content:
            continue
        turns.append(
            {
                "turn_id": f"{case.get('episode_id', i)}:{role}:{i}",
                "role": role,
                "content": content,
            }
        )
    return {
        "id": str(case.get("episode_id") or case.get("id") or len(turns)),
        "session_id": str(case.get("episode_id") or case.get("id") or len(turns)),
        "dataset": str(case.get("dataset") or "tau2"),
        "split": "eval",
        "turns": turns,
        "metadata": {
            "domain": case.get("domain") or (case.get("metadata") or {}).get("domain"),
            "tau2_task_id": case.get("tau2_task_id") or case.get("task_id"),
        },
    }


def select_cases(cases: list[dict[str, Any]], max_cases: int) -> list[dict[str, Any]]:
    """Keep the first N cases that have at least one user turn (stable order)."""

    selected: list[dict[str, Any]] = []
    for case in cases:
        rec = case_to_record(case)
        if any(t["role"] == "user" for t in rec["turns"]):
            selected.append(case)
        if max_cases and len(selected) >= max_cases:
            break
    return selected


def manual_to_persona(manual: dict[str, Any]) -> dict[str, Any]:
    """Convert a cue manual into the RealUserSim sidecar profile shape."""

    profile = to_profile(manual)
    general_text, specific_text, style_text = split_manual_targets(manual)
    general = parse_manual(general_text)
    specific = parse_manual(specific_text)
    style = parse_manual(style_text)
    return {
        "demographics": {},
        **profile,
        "manual_sections": render_dual_manual(general, specific, style),
        "manual_text": canonicalize_profile(manual),
    }


def write_persona_sidecar(path: Path, profiles: list[dict[str, Any]], *, seed: int = 0) -> None:
    """Write paired + sample_shuffled rows keyed for tau2 runner lookup."""

    from cue_training.evaluation.common.shuffle import build_shuffle_map, domain_of

    path.parent.mkdir(parents=True, exist_ok=True)
    order = [p["id"] for p in profiles]
    by_id = {p["id"]: p for p in profiles}
    domains = {
        eid: domain_of(eid, record=by_id[eid].get("provenance") or by_id[eid])
        for eid in order
    }
    shuffle_map = build_shuffle_map(order, domains=domains, seed=seed)
    with path.open("w", encoding="utf-8") as handle:
        for profile in profiles:
            eid = profile["id"]
            persona = manual_to_persona(profile["persona_manual"])
            origin_id = shuffle_map[eid]
            shuffled_persona = manual_to_persona(by_id[origin_id]["persona_manual"])
            for arm, persona_obj, origin in (
                ("paired", persona, eid),
                ("sample_shuffled", shuffled_persona, origin_id),
            ):
                handle.write(
                    json.dumps(
                        {
                            "episode_id": eid,
                            "variant": REALUSERSIM_VARIANT,
                            "arm": arm,
                            "persona": persona_obj,
                            "persona_text": persona_obj.get("manual_text") or "",
                            "shuffled_episode_id": shuffle_map[eid],
                            "persona_origin_episode_id": origin,
                            "provenance": profile.get("provenance") or {},
                        },
                        ensure_ascii=False,
                    )
                    + "\n"
                )


def _exclude_for_rollout_sim(rollout_sim: str) -> str | None:
    if "llama" in rollout_sim.lower():
        return "Llama"
    if "gpt" in rollout_sim.lower():
        return "gpt"
    return None


def generate_profiles_for_arm(
    *,
    cases: list[dict[str, Any]],
    proposer: str,
    rollout_sim: str,
    sim_models: list[str],
    sim_api_bases: dict[str, str | None],
    proposer_api_base: str | None,
    out_path: Path,
    case_workers: int = 8,
    sim_workers: int = 8,
    max_user_turns: int = 15,
    style_commands: int = 0,
    router: ReplicaRouter | None = None,
) -> Path:
    """Build one persona sidecar; cases are processed concurrently within this proposer."""

    records = [case_to_record(c) for c in cases]
    exclude = _exclude_for_rollout_sim(rollout_sim)
    router = router or build_default_router()
    workers = max(1, int(case_workers))
    profiles: list[dict[str, Any]] = []
    done = 0

    def _one(rec: dict[str, Any]) -> dict[str, Any] | None:
        pool = [r for r in records if r["id"] != rec["id"]]
        negs = pool[:5]
        try:
            built = build_manual_for_session(
                rec,
                negatives=negs,
                sim_models=sim_models,
                proposer=proposer,
                sim_api_bases=sim_api_bases,
                proposer_api_base=proposer_api_base,
                exclude_sim=exclude,
                router=router,
                sim_workers=sim_workers,
                max_user_turns=max_user_turns,
            )
            if style_commands > 0:
                style = extract_style_commands(
                    turns=list(rec.get("turns") or []),
                    model=proposer,
                    n_style=style_commands,
                    api_base=proposer_api_base,
                )
                manual = dict(built["persona_manual"])
                manual["commands"] = list(manual.get("commands") or []) + style
                built["persona_manual"] = validate_manual(manual)
            return built
        except Exception as exc:  # noqa: BLE001
            warn(TAG, f"profile failed id={rec['id']} proposer={proposer}: {exc}")
            return None

    log(TAG, f"profiles proposer={proposer} rollout={rollout_sim} n={len(records)} workers={workers}")
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = [pool.submit(_one, rec) for rec in records]
        for fut in as_completed(futures):
            built = fut.result()
            done += 1
            if built is not None:
                profiles.append(built)
            if done % 10 == 0 or done == len(records):
                log(TAG, f"{proposer}@{rollout_sim}: {done}/{len(records)}")

    by_id = {p["id"]: p for p in profiles}
    ordered = [by_id[r["id"]] for r in records if r["id"] in by_id]
    from cue_training.evaluation.common.shuffle import DEFAULT_SHUFFLE_SEED

    write_persona_sidecar(out_path, ordered, seed=DEFAULT_SHUFFLE_SEED)
    log(TAG, f"wrote {len(ordered)} profiles -> {out_path}")
    return out_path


def _simarena_cases(path: str | Path, domain: str) -> list[dict[str, Any]]:
    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    annotations = raw if isinstance(raw, list) else list(raw.values())
    cases = []
    for annotation in annotations:
        users = annotation.get("user_queries") or []
        assistants = (
            annotation.get("ai_responses")
            or annotation.get("assistant_responses")
            or []
        )
        conversation: list[dict[str, str]] = []
        for index, user in enumerate(users):
            conversation.append({"role": "user", "content": str(user)})
            if index < len(assistants):
                conversation.append(
                    {"role": "assistant", "content": str(assistants[index])}
                )
        if not conversation:
            continue
        if domain == "math":
            episode_id = (
                f"{annotation.get('model')}_{annotation.get('workerId')}_"
                f"{annotation.get('problem_id')}"
            )
        else:
            episode_id = (
                f"{annotation.get('model')}_{annotation.get('workerId')}_"
                f"{annotation.get('document_type')}_{annotation.get('intent')}"
            )
        cases.append(
            {
                "episode_id": episode_id,
                "domain": domain,
                "dataset": "simarena",
                "real_conversation": conversation,
            }
        )
    return cases


def _prism_cases(path: str | Path) -> list[dict[str, Any]]:
    """PRISM MirrorBench normalized.jsonl — same episode/conversation shape as tau2."""

    return load_tau2_cases(path)


def build_eval_profiles(
    *,
    config: str | Path,
    benchmark: str,
    input_path: str | Path,
    output: str | Path,
    rollout_sim: str,
    domain: str = "",
    max_cases: int = 0,
    case_workers: int = 8,
) -> Path:
    cfg = load_config(config)
    if benchmark == "tau2":
        cases = load_tau2_cases(input_path)
    elif benchmark == "simarena":
        if domain not in {"writing", "math"}:
            raise ValueError("SimulatorArena domain must be writing or math")
        cases = _simarena_cases(input_path, domain)
    elif benchmark == "prism":
        cases = _prism_cases(input_path)
    else:
        raise ValueError("benchmark must be tau2, simarena, or prism")
    cases = select_cases(cases, max_cases)
    return generate_profiles_for_arm(
        cases=cases,
        proposer=str(get(cfg, "proposer_model", "gpt-5.4-mini")),
        rollout_sim=rollout_sim,
        sim_models=list(get(cfg, "sim_models", [])),
        sim_api_bases=dict(get(cfg, "sim_api_bases", {})),
        proposer_api_base=get(cfg, "proposer_api_base"),
        out_path=Path(output),
        case_workers=case_workers,
        sim_workers=int(get(cfg, "sim_workers", 8)),
        max_user_turns=int(get(cfg, "max_user_turns", 15)),
        style_commands=int(get(cfg, "style_commands", 0)),
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Build data annotation eval persona sidecars.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--benchmark", choices=("tau2", "simarena", "prism"), required=True)
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--rollout_sim", required=True)
    parser.add_argument("--domain", choices=("writing", "math"), default=None)
    parser.add_argument("--max_cases", type=int, default=0)
    parser.add_argument("--case_workers", type=int, default=8)
    args = parser.parse_args()
    build_eval_profiles(
        config=args.config,
        benchmark=args.benchmark,
        input_path=args.input,
        output=args.output,
        rollout_sim=args.rollout_sim,
        domain=args.domain or "",
        max_cases=args.max_cases,
        case_workers=args.case_workers,
    )


if __name__ == "__main__":
    main()
