"""data annotation: simulator-contrastive training-set construction."""

from __future__ import annotations

import argparse
import json
import random
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

from tqdm import tqdm

from cue_training.augment import select_turns_for_persona
from cue_training.data.schema import validate_manual, validate_record
from cue_training.data.streaming import discover_jsonl_files, iter_jsonl, write_jsonl_atomic, write_streaming_index
from cue_training.preprocessing.delexicalize import delexicalize_manual
from cue_training.preprocessing.dispatch import ReplicaRouter, build_default_router
from cue_training.preprocessing.gates import ManualGateError, gate_manual, ground_evidence_turn_ids
from cue_training.preprocessing.llm import complete, complete_json
from cue_training.preprocessing.prompts import build_extraction_messages
from cue_training.utils.config import get, load_config
from cue_training.utils.log import log, warn

TAG = "data_annotation"

DEFAULT_SIMS = [
    "meta-llama/Llama-3.1-8B-Instruct",
    "Qwen/Qwen3-8B",
    "gpt-5.4-mini",
    "claude-haiku-4-5-20251001",
]


def _user_positions(turns: list[dict[str, Any]]) -> list[int]:
    return [i for i, t in enumerate(turns) if t.get("role") == "user"]


def _prefix_until(turns: list[dict[str, Any]], user_idx: int) -> list[dict[str, str]]:
    return [
        {"role": t["role"], "content": t["content"]}
        for t in turns[:user_idx]
        if t.get("role") in {"user", "assistant", "system"}
    ]


def simulate_user_turn(
    *,
    model: str,
    history: list[dict[str, str]],
    api_base: str | None,
    router: ReplicaRouter | None = None,
    temperature: float = 0.7,
    max_tokens: int = 128,
    max_input_tokens: int = 7800,
) -> str:
    """One-step counterfactual user reply; truncates history to fit local vLLM context."""

    messages = [
        {
            "role": "system",
            "content": (
                "You are simulating a real user in a multi-turn conversation. "
                "Reply as the user only, briefly, with no assistant meta-commentary."
            ),
        }
    ]
    for turn in history:
        role = "assistant" if turn["role"] in {"assistant", "system"} else "user"
        content = str(turn.get("content") or "")
        if len(content) > 2000:
            content = content[:800] + "\n…\n" + content[-800:]
        messages.append({"role": role, "content": content})
    messages.append(
        {
            "role": "user",
            "content": "Continue as the user with your next message only.",
        }
    )

    def _call(base: str | None, input_budget: int) -> str:
        return complete(
            model=model,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
            api_base=base,
            max_input_tokens=input_budget,
        )

    budgets = [max_input_tokens, 6000, 4000, 2500]
    last_exc: Exception | None = None
    for budget in budgets:
        try:
            if router is not None:
                with router.api_base_for(model, api_base) as resolved_base:
                    return _call(resolved_base, budget)
            return _call(api_base, budget)
        except Exception as exc:  # noqa: BLE001
            last_exc = exc
            err = str(exc)
            if "ContextWindowExceeded" not in err and "maximum context length" not in err:
                raise
            warn(TAG, f"{model} context overflow at budget={budget}; retrying smaller")
    assert last_exc is not None
    raise last_exc


def select_cohort(
    records: list[dict[str, Any]],
    *,
    cap: int,
    seed: int,
) -> list[dict[str, Any]]:
    if len(records) <= cap:
        return records
    rng = random.Random(seed)
    buckets: dict[str, list[dict[str, Any]]] = {}
    for rec in records:
        n = sum(1 for t in rec["turns"] if t.get("role") == "user")
        key = "1" if n <= 1 else "2-4" if n <= 4 else "5-10" if n <= 10 else "11+"
        buckets.setdefault(key, []).append(rec)
    per = max(1, cap // max(1, len(buckets)))
    chosen: list[dict[str, Any]] = []
    for items in buckets.values():
        rng.shuffle(items)
        chosen.extend(items[:per])
    rng.shuffle(chosen)
    return chosen[:cap]


def _run_sim_jobs(
    *,
    turns: list[dict[str, Any]],
    user_idxs: list[int],
    active_sims: list[str],
    sim_api_bases: dict[str, str | None],
    router: ReplicaRouter | None,
    sim_workers: int,
) -> dict[str, list[str]]:
    """Fan out all (user_turn, simulator) calls; preserve turn order per model."""

    jobs: list[tuple[int, int, str, list[dict[str, str]]]] = []
    for turn_i, ui in enumerate(user_idxs):
        history = _prefix_until(turns, ui)
        for model in active_sims:
            jobs.append((turn_i, ui, model, history))

    n_turns = len(user_idxs)
    # slot[model][turn_i] = response
    slots: dict[str, list[str | None]] = {m: [None] * n_turns for m in active_sims}
    workers = max(1, min(sim_workers, len(jobs) or 1))

    def _one(job: tuple[int, int, str, list[dict[str, str]]]) -> tuple[int, str, str]:
        turn_i, _ui, model, history = job
        text = simulate_user_turn(
            model=model,
            history=history,
            api_base=sim_api_bases.get(model),
            router=router,
        )
        return turn_i, model, text

    if workers == 1 or len(jobs) <= 1:
        for job in jobs:
            turn_i, model, text = _one(job)
            slots[model][turn_i] = text
    else:
        with ThreadPoolExecutor(max_workers=workers) as pool:
            futs = [pool.submit(_one, job) for job in jobs]
            for fut in as_completed(futs):
                turn_i, model, text = fut.result()
                slots[model][turn_i] = text

    return {m: [t if t is not None else "" for t in slots[m]] for m in active_sims}


def build_manual_for_session(
    record: dict[str, Any],
    *,
    negatives: list[dict[str, Any]],
    sim_models: list[str],
    proposer: str,
    sim_api_bases: dict[str, str | None],
    proposer_api_base: str | None,
    max_user_turns: int = 15,
    style_commands: int = 0,
    exclude_sim: str | None = None,
    router: ReplicaRouter | None = None,
    sim_workers: int = 8,
) -> dict[str, Any]:
    turns = select_turns_for_persona(record["turns"], max_user_turns=max_user_turns)
    user_idxs = _user_positions(turns)
    active_sims = [m for m in sim_models if exclude_sim is None or exclude_sim not in m]
    sim_responses = _run_sim_jobs(
        turns=turns,
        user_idxs=user_idxs,
        active_sims=active_sims,
        sim_api_bases=sim_api_bases,
        router=router,
        sim_workers=sim_workers,
    )

    messages = build_extraction_messages(
        target_turns=turns,
        human_negatives=[n["turns"] for n in negatives],
        simulator_turns=sim_responses,
    )
    gate_notes: list[str] = []
    manual: dict[str, Any] | None = None
    # Propose → ground evidence → delex → gate; up to 3 rewrite passes on gate failure.
    gate_retries = 3
    for attempt in range(1 + gate_retries):
        raw = complete_json(
            model=proposer,
            messages=messages,
            temperature=0.2 if attempt == 0 else 0.35,
            max_tokens=2500,
            api_base=proposer_api_base,
            response_format={"type": "json_object"},
            max_input_tokens=100_000,
            repair_attempts=2,
        )
        candidate = delexicalize_manual(
            ground_evidence_turn_ids(validate_manual(raw), turns)
        )
        try:
            gate_manual(
                candidate,
                turns=turns,
                negatives=[n["turns"] for n in negatives],
            )
            manual = candidate
            break
        except ManualGateError as exc:
            gate_notes.append(str(exc))
            warn(TAG, f"manual gate failed id={record.get('id')} attempt={attempt}: {exc}")
            messages = messages + [
                {
                    "role": "user",
                    "content": (
                        f"Your previous manual failed validation: {exc}. "
                        "Rewrite the full 10-command JSON. Cite real TARGET turn_ids, "
                        "remove task content, and make at least 2 human_contrast commands "
                        "encode habits that differentiate TARGET from the human negatives "
                        "(lag, escalation, pushback, impatience, try-then-report, smalltalk, "
                        "playful, acceptance-after-refusal) — not soft closers shared by all humans."
                    ),
                }
            ]
    if manual is None:
        raise ManualGateError("; ".join(gate_notes) or "manual failed gates")
    if style_commands:
        from cue_training.preprocessing.style_augment import extract_style_commands

        manual = {
            **manual,
            "commands": [
                *(manual.get("commands") or []),
                *extract_style_commands(
                    turns=turns,
                    model=proposer,
                    n_style=style_commands,
                    api_base=proposer_api_base,
                ),
            ],
        }
    return {
        **record,
        "turns": record["turns"],
        "persona_manual": manual,
        "provenance": {
            **(record.get("provenance") or {}),
            "proposer": proposer,
            "sim_models": active_sims,
            "human_negative_ids": [n["id"] for n in negatives],
            "exclude_sim": exclude_sim,
            "max_user_turns": max_user_turns,
            "selected_turn_ids": [t.get("turn_id") for t in turns if t.get("role") == "user"],
            "gate_notes": gate_notes,
        },
    }


def _process_dataset(
    *,
    path: Path,
    dataset: str,
    split: str,
    output_root: Path,
    cohort: list[dict[str, Any]],
    seed: int,
    n_neg: int,
    sim_models: list[str],
    sim_api_bases: dict[str, str | None],
    proposer: str,
    proposer_api_base: str | None,
    max_user_turns: int,
    router: ReplicaRouter | None,
    session_workers: int,
    sim_workers: int,
    style_commands: int,
) -> tuple[Path, int]:
    out_dir = output_root / dataset
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{split}.jsonl"
    err_path = out_dir / f"{split}.error.jsonl"
    err_lock = threading.Lock()
    outputs: list[dict[str, Any] | None] = [None] * len(cohort)
    workers = max(1, min(session_workers, len(cohort) or 1))

    def _one(i: int, rec: dict[str, Any]) -> tuple[int, dict[str, Any] | None, str | None]:
        pool = [r for r in cohort if r["id"] != rec["id"]]
        rng = random.Random(seed + i)
        negs = rng.sample(pool, k=min(n_neg, len(pool))) if pool else []
        try:
            built = build_manual_for_session(
                rec,
                negatives=negs,
                sim_models=sim_models,
                proposer=proposer,
                sim_api_bases=sim_api_bases,
                proposer_api_base=proposer_api_base,
                max_user_turns=max_user_turns,
                style_commands=style_commands,
                router=router,
                sim_workers=sim_workers,
            )
            return i, built, None
        except Exception as exc:  # noqa: BLE001
            return i, None, str(exc)

    with ThreadPoolExecutor(max_workers=workers) as pool:
        futs = [pool.submit(_one, i, rec) for i, rec in enumerate(cohort)]
        for fut in tqdm(
            as_completed(futs),
            total=len(futs),
            desc=f"data_annotation:{dataset}",
            unit="session",
        ):
            i, built, err = fut.result()
            if built is not None:
                outputs[i] = built
            else:
                warn(TAG, f"failed session={cohort[i]['id']}: {err}")
                with err_lock:
                    with err_path.open("a", encoding="utf-8") as handle:
                        handle.write(json.dumps({"id": cohort[i]["id"], "error": err}) + "\n")

    written = [o for o in outputs if o is not None]
    write_jsonl_atomic(out_path, written)
    log(TAG, f"wrote {len(written)}/{len(cohort)} -> {out_path}")
    return out_path, len(written)


def _count_jsonl_lines(path: Path) -> int:
    if not path.is_file() or path.stat().st_size == 0:
        return 0
    n = 0
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                n += 1
    return n


def run_data_annotation(cfg: dict[str, Any]) -> Path:
    data_root = Path(get(cfg, "data_root"))
    output_root = Path(get(cfg, "output_root", "data/profiled"))
    output_root.mkdir(parents=True, exist_ok=True)
    split = get(cfg, "split", "train")
    cap = int(get(cfg, "per_dataset_cap", 10_000))
    seed = int(get(cfg, "seed", 0))
    n_neg = int(get(cfg, "num_human_negatives", 5))
    max_user_turns = int(get(cfg, "max_user_turns", 15))
    proposer = get(cfg, "proposer_model", "gemini/gemini-3.1-pro-preview")
    proposer_api_base = get(cfg, "proposer_api_base")
    sim_models = list(get(cfg, "sim_models", DEFAULT_SIMS))
    sim_api_bases = dict(get(cfg, "sim_api_bases", {}))
    max_sessions = int(get(cfg, "max_sessions", 0))  # 0 = all selected
    session_workers = int(get(cfg, "session_workers", 8))
    sim_workers = int(get(cfg, "sim_workers", 8))
    skip_existing = bool(get(cfg, "skip_existing", True))
    use_replica_lb = bool(get(cfg, "least_inflight_replicas", True))
    router = build_default_router() if use_replica_lb else None

    files = discover_jsonl_files(data_root, split=split)
    if not files:
        raise FileNotFoundError(f"No {split}.jsonl under {data_root}")

    log(
        TAG,
        f"parallelism session_workers={session_workers} sim_workers={sim_workers} "
        f"proposer={proposer} skip_existing={skip_existing}",
    )
    written_files: list[Path] = []
    total = 0
    for path in files:
        dataset = path.parent.name
        out_path = output_root / dataset / f"{split}.jsonl"
        if skip_existing:
            existing_n = _count_jsonl_lines(out_path)
            if existing_n > 0:
                log(TAG, f"skip done dataset={dataset} n={existing_n} -> {out_path}")
                written_files.append(out_path)
                total += existing_n
                continue
        log(TAG, f"processing dataset={dataset} file={path}")
        records: list[dict[str, Any]] = []
        skipped = 0
        for raw in iter_jsonl(path, validate=False):
            try:
                rec = validate_record(raw)
            except (ValueError, TypeError) as exc:
                skipped += 1
                if skipped <= 5:
                    warn(
                        TAG,
                        f"skip invalid record dataset={dataset} "
                        f"id={raw.get('id') if isinstance(raw, dict) else '?'} err={exc}",
                    )
                continue
            rec["dataset"] = dataset
            rec["split"] = split
            records.append(rec)
        if skipped:
            warn(TAG, f"skipped {skipped} invalid records in dataset={dataset}")
        if not records:
            warn(TAG, f"no valid records in dataset={dataset}; skipping")
            continue
        cohort = select_cohort(records, cap=cap, seed=seed + hash(dataset) % 10_000)
        if max_sessions:
            cohort = cohort[:max_sessions]
        out_path, n = _process_dataset(
            path=path,
            dataset=dataset,
            split=split,
            output_root=output_root,
            cohort=cohort,
            seed=seed,
            n_neg=n_neg,
            sim_models=sim_models,
            sim_api_bases=sim_api_bases,
            proposer=proposer,
            proposer_api_base=proposer_api_base,
            max_user_turns=max_user_turns,
            style_commands=int(get(cfg, "style_commands", 0)),
            router=router,
            session_workers=session_workers,
            sim_workers=sim_workers,
        )
        written_files.append(out_path)
        total += n

    write_streaming_index(
        output_root,
        written_files,
        extra={
            "pipeline": "data_annotation",
            "proposer": proposer,
            "sim_models": sim_models,
            "session_workers": session_workers,
            "sim_workers": sim_workers,
        },
    )
    log(TAG, f"done total={total} output_root={output_root}")
    return output_root


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="data annotation dual-contrast profile construction.")
    p.add_argument("--config", required=True)
    p.add_argument("--data_root", default=None)
    p.add_argument("--output_root", default=None)
    p.add_argument(
        "--skip_existing",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Skip datasets whose output JSONL already has rows (default: config/true).",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)
    if args.data_root:
        cfg["data_root"] = args.data_root
    if args.output_root:
        cfg["output_root"] = args.output_root
    if args.skip_existing is not None:
        cfg["skip_existing"] = args.skip_existing
    run_data_annotation(cfg)


if __name__ == "__main__":
    main()
