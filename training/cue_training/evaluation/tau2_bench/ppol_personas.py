"""Generate a tau2 persona sidecar for the PPol baseline (faithful, as_is only).

Unlike the other persona baselines (realusersim/usp), PPol does NOT
distill a persona from each episode's real conversation and has NO paired/shuffled
arms. Its personas are produced by the evolved generator ``G(c, D, N)`` conditioned
on the task scenario ``c``, so they are grouped BY TASK:

  - group the eval episodes by ``tau2_task_id`` (the 3 tau-usi lines of a task
    share one scenario),
  - run ``G(c, D, N)`` ONCE per task with paper final ``N`` (default 10; curriculum
    peaks at 10), using ``format_user_scenario_c`` for ``c`` when the tau2 task is
    available,
  - assign one distinct persona per line (index within the task group), so cross-
    task shuffling -- which could leak another task's facts -- never happens.

Writes ``{variant: "baseline:ppol", arm: "as_is", episode_id, persona}`` rows read
by the runner's ``--personas`` (``read_persona_map``) and injected into the PPol
simulator's ``_persona``.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

from cue_training.evaluation.common.io import read_jsonl, write_jsonl
from cue_training.evaluation.tau2_bench.data import scenario_only
from cue_training.runlog.log import log, warn

# Paper final N (matches PersonaPoliciesConfig.n_personas after curriculum 5→8→10).
DEFAULT_N_PERSONAS = 10
# Paper generator / mutator default (LiteLLM id).
DEFAULT_GEN_MODEL = "openrouter/google/gemini-3-flash-preview"

_VENDOR_ROOT = (
    Path(__file__).resolve().parents[2]
    / "baselines"
    / "ppol"
    / "external"
    / "persona-policies"
)


def _group_by_task(cases: list[dict[str, Any]]) -> "OrderedDict[str, list[dict[str, Any]]]":
    """Group unique episodes by tau2_task_id, preserving first-seen order."""

    groups: OrderedDict[str, list[dict[str, Any]]] = OrderedDict()
    seen: set[str] = set()
    for c in cases:
        eid = str(c.get("episode_id"))
        if eid in seen:
            continue
        seen.add(eid)
        tid = str(c.get("tau2_task_id") if c.get("tau2_task_id") is not None else eid)
        groups.setdefault(tid, []).append(c)
    return groups


def _ensure_vendor_on_path() -> None:
    root = str(_VENDOR_ROOT)
    if root not in sys.path:
        sys.path.insert(0, root)


def _load_tau2_task(
    tau2_path: Path | None,
    domain: str,
    task_id: str,
) -> dict[str, Any] | None:
    if tau2_path is None or not domain or task_id is None or task_id == "":
        return None
    path = Path(tau2_path) / "data" / "tau2" / "domains" / domain / "tasks.json"
    if not path.is_file():
        return None
    try:
        tasks = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:  # noqa: BLE001
        warn("tau2:ppol", f"failed to read {path}: {exc}")
        return None
    want = str(task_id)
    for task in tasks:
        if not isinstance(task, dict):
            continue
        if str(task.get("_combined_id", task.get("id"))) == want:
            return task
        if str(task.get("id")) == want:
            return task
    return None


def task_context_c(
    case: dict[str, Any],
    *,
    tau2_path: Path | None,
) -> str:
    """Paper ``c``: ``format_user_scenario_c(task)``, with scenario_only fallback."""

    domain = str(case.get("tau2_domain") or case.get("domain") or "")
    tid = case.get("tau2_task_id")
    task = _load_tau2_task(tau2_path, domain, str(tid) if tid is not None else "")
    if task is not None:
        _ensure_vendor_on_path()
        from persona_policies.tau_train_context import format_user_scenario_c

        return format_user_scenario_c(task)
    fallback = scenario_only(case.get("task_description") or "") or (
        case.get("task_description") or ""
    )
    if not str(fallback).strip():
        warn(
            "tau2:ppol",
            f"no format_user_scenario_c for episode={case.get('episode_id')} "
            f"(domain={domain!r} task_id={tid!r}); using empty c",
        )
    return str(fallback)


def run(args: argparse.Namespace) -> None:
    from cue_training.baselines.ppol.generate import generate_personas

    cases = read_jsonl(str(args.cases))
    groups = _group_by_task(cases)
    if args.limit:
        groups = OrderedDict(list(groups.items())[: args.limit])

    tau2_path = Path(args.tau2_path) if getattr(args, "tau2_path", None) else None
    gen_model = (
        getattr(args, "gen_model", None)
        or getattr(args, "sim_model", None)
        or DEFAULT_GEN_MODEL
    )
    api_key_env = getattr(args, "api_key_env", None) or "OPENAI_API_KEY"
    # Gemini via Google AI Studio uses GOOGLE_API_KEY when the litellm id is gemini/*.
    if str(gen_model).startswith("gemini/") and api_key_env == "OPENAI_API_KEY":
        api_key_env = "GOOGLE_API_KEY"
    if str(gen_model).startswith("openrouter/") and api_key_env == "OPENAI_API_KEY":
        api_key_env = "OPENROUTER_API_KEY"

    def _personas_for_task(item: tuple[str, list[dict[str, Any]]]) -> tuple[str, list[str]]:
        tid, members = item
        scenario = task_context_c(members[0], tau2_path=tau2_path)
        # Paper N, but never fewer than the number of tau-usi lines for this task.
        n = max(int(args.n), len(members))
        try:
            personas = generate_personas(
                scenario,
                n,
                best_program_path=args.best_program,
                model=gen_model,
                api_base=args.api_base,
                api_key_env=api_key_env,
            )
        except Exception as exc:  # noqa: BLE001 - one bad task shouldn't kill the sidecar
            warn("tau2:ppol", f"task {tid}: generation failed ({exc}); leaving to default persona")
            personas = []
        return tid, personas

    with ThreadPoolExecutor(max_workers=max(1, args.workers)) as ex:
        results = dict(ex.map(_personas_for_task, groups.items()))

    rng = random.Random(args.seed)
    rows: list[dict[str, Any]] = []
    n_assigned = 0
    for tid, members in groups.items():
        personas = list(results.get(tid) or [])
        if personas:
            rng.shuffle(personas)  # de-correlate persona index from line order (seeded)
        ordered = sorted(members, key=lambda c: str(c.get("episode_id")))
        for i, case in enumerate(ordered):
            if not personas:
                continue  # no row -> PPol simulator falls back to its default persona
            persona = personas[i % len(personas)]
            n_assigned += 1
            rows.append({
                "variant": "baseline:ppol", "arm": "as_is",
                "episode_id": str(case.get("episode_id")), "persona": persona,
            })

    write_jsonl(str(args.out), rows)
    log(
        "tau2:ppol",
        f"{len(groups)} tasks; assigned {n_assigned} personas (N={args.n}, gen={gen_model}); "
        f"wrote {len(rows)} rows -> {args.out}",
    )


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Generate a PPol persona sidecar (as_is, per-task G(c,D,N)).")
    p.add_argument("--cases", required=True, help="cases.paired.jsonl (episode_id + tau2_task_id + task_description)")
    p.add_argument("--out", required=True, help="Output PPol personas sidecar JSONL.")
    p.add_argument("--best_program", required=True, help="Path to the evolved best_program.py (G(c,D,N)).")
    p.add_argument(
        "--n",
        type=int,
        default=DEFAULT_N_PERSONAS,
        help=f"Personas per task for G(c,D,N) (default {DEFAULT_N_PERSONAS} = paper final N).",
    )
    p.add_argument(
        "--gen_model",
        default=DEFAULT_GEN_MODEL,
        help="Persona generator LLM (litellm id); paper default Gemini Flash.",
    )
    p.add_argument(
        "--sim_model",
        default=None,
        help="Deprecated alias for --gen_model (generator, not the user-sim).",
    )
    p.add_argument("--tau2_path", default=None, help="tau2-bench checkout (for format_user_scenario_c).")
    p.add_argument("--api_base", default=None)
    p.add_argument("--api_key_env", default="OPENAI_API_KEY")
    p.add_argument("--workers", type=int, default=8, help="Task-level generation concurrency.")
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--seed", type=int, default=0)
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    run(parse_args(argv))


if __name__ == "__main__":
    main()
