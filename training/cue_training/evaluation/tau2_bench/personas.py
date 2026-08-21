"""Generate a tau2 persona sidecar for the persona baselines (realusersim/usp).

tau2 has no per-episode persona source (unlike SimulatorArena's profiles), so without a
sidecar the paired vs sample_shuffled arms don't differ. This extracts, per tau_usi eval
episode, a persona from that episode's ``real_conversation`` IN EACH BASELINE'S OWN FORMAT
(so each stays a faithful baseline), then writes ``(variant, arm, episode_id) -> persona``
rows consumed by the runner's ``--personas`` (``read_persona_map``), overriding native pool
sampling per episode:
  - ``paired``          = the episode's own extracted persona (1:1 with the evaluated task)
  - ``sample_shuffled`` = a seeded shuffle to another episode's persona (the mismatch control)
  - ``sample_diverse``  = USP Diverse Profile Sampling over LMSYS-USP (USP only)

Native formats (matching each simulator's injected attribute):
  - realusersim (``_profile``)      -> RealUserSim profile dict (Command+Example + demographics)
  - usp (``_profile_text``)         -> USP implicit-profile text
"""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

from cue_training.evaluation.common.io import read_jsonl, write_jsonl
from cue_training.runlog.log import log, warn

PERSONA_BASELINES = ("usp", "realusersim_paired_noex")

_RUS_DEFAULT = {
    "demographics": {
        "age": {"value": None, "source": None, "confidence": None},
        "occupation": {"value": "everyday user", "source": "inferred", "confidence": 0.4},
    },
    "additional_background": None,
    "command_examples": [
        {
            "command": "Be terse and get to the point.",
            "examples": ["can you fix this?", "still broken"],
        },
        {
            "command": "Skip pleasantries.",
            "examples": ["ok now add the tests"],
        },
        {
            "command": "Push back when the assistant misunderstands.",
            "examples": ["no thats not what i asked"],
        },
    ],
    "commands": [
        "Be terse and get to the point.",
        "Skip pleasantries.",
        "Push back when the assistant misunderstands.",
    ],
    "examples": ["can you fix this?", "still broken", "ok now add the tests"],
}


def _user_turns_transcript(conversation: list[dict[str, Any]]) -> str:
    return "\n".join(f"- {t.get('content', '')}" for t in conversation if t.get("role") == "user").strip()


def _extract_realusersim(conv: list[dict[str, Any]], **kw: Any) -> dict[str, Any]:
    """RealUserSim profile dict (injected into _profile); GPT-4o paper extractor."""

    from cue_training.baselines.realusersim.profiles import DEFAULT_PROFILER_MODEL, extract_profile_from_user_text

    text = _user_turns_transcript(conv)
    model = kw.get("model") or DEFAULT_PROFILER_MODEL
    # Paper profiler is GPT-4o; force it for RealUserSim even if the rollout sim is another model.
    if "gpt-4o" not in str(model).lower():
        model = DEFAULT_PROFILER_MODEL
    parsed = extract_profile_from_user_text(
        text,
        model=model,
        api_key_env=kw.get("api_key_env") or "OPENAI_API_KEY",
        api_base=kw.get("api_base"),
        include_examples=bool(kw.get("include_examples", True)),
    ) if text else None
    return parsed if isinstance(parsed, dict) else dict(_RUS_DEFAULT)


def _extract_realusersim_paired_noex(conv: list[dict[str, Any]], **kw: Any) -> dict[str, Any]:
    """Paired RealUserSim extract with examples stripped (eval leakage control)."""

    return _extract_realusersim(conv, **{**kw, "include_examples": False})


def _extract_usp(conv: list[dict[str, Any]], **kw: Any) -> dict[str, Any]:
    """USP implicit-profile dict (rendered to text at write time for ``_profile_text``)."""
    from cue_training.baselines.usp.extractor import _PROFILE_PROMPT
    from cue_training.baselines.usp.sampler import _DEFAULT_PROFILE

    text = _user_turns_transcript(conv)
    parsed = _chat_json(_PROFILE_PROMPT, text, **kw) if text else None
    return parsed if isinstance(parsed, dict) else dict(_DEFAULT_PROFILE)


def _chat_json(system: str, transcript: str, *, model: str, api_key_env: str,
               api_base: str | None) -> dict | None:
    from cue_training.baselines.common.llm import chat, parse_json_block

    try:
        raw = chat(
            [{"role": "system", "content": system},
             {"role": "user", "content": f"Conversation user turns:\n{transcript}"}],
            model=model, api_key_env=api_key_env, api_base=api_base, temperature=0.3, max_tokens=600,
        )
    except Exception as exc:  # noqa: BLE001
        warn("tau2:personas", f"extraction failed: {exc}")
        return None
    parsed = parse_json_block(raw)
    return parsed if isinstance(parsed, dict) else None


_EXTRACTORS = {
    "realusersim": _extract_realusersim,
    "realusersim_paired_noex": _extract_realusersim_paired_noex,
    "usp": _extract_usp,
}

# Sidecar ``variant`` matches the recipe harness variant (baseline:<method>).
_HARNESS_VARIANT: dict[str, str] = {}


def _usp_persona_value(profile: dict[str, Any]) -> str:
    from cue_training.baselines.usp.extractor import profile_text

    return profile_text(profile)


def sidecar_covers_arms(path: str | Path, variant: str, arms: list[str] | tuple[str, ...]) -> bool:
    """True if ``path`` already has rows for ``variant`` covering every persona arm in ``arms``."""

    needed = {a for a in arms if a in ("paired", "sample_shuffled", "sample_diverse")}
    if not needed:
        return True
    p = Path(path)
    if not p.is_file() or p.stat().st_size == 0:
        return False
    found: set[str] = set()
    for row in read_jsonl(str(p)):
        if row.get("variant") == variant and row.get("arm"):
            found.add(str(row["arm"]))
            if needed <= found:
                return True
    return needed <= found


def run(args: argparse.Namespace) -> None:
    cases = read_jsonl(str(args.cases))
    episodes: list[tuple[str, list[dict[str, Any]]]] = []
    seen: set[str] = set()
    for c in cases:  # cases.paired.jsonl has one row per episode; dedupe defensively
        eid = str(c.get("episode_id"))
        if eid in seen:
            continue
        seen.add(eid)
        episodes.append((eid, c.get("real_conversation") or []))
    if args.limit:
        episodes = episodes[: args.limit]

    variants = [v for v in PERSONA_BASELINES if v in set(args.variants)] or list(PERSONA_BASELINES)
    kw = {"model": args.sim_model, "api_key_env": args.api_key_env, "api_base": args.api_base}

    # Extract per (variant, episode) in each baseline's native format, concurrently.
    tasks = [(v, eid, conv) for v in variants for (eid, conv) in episodes]

    def _do(task: tuple[str, str, list[dict[str, Any]]]):
        v, eid, conv = task
        return v, eid, _EXTRACTORS[v](conv, **kw)

    with ThreadPoolExecutor(max_workers=max(1, args.batch_size)) as ex:
        results = list(ex.map(_do, tasks))
    by_variant: dict[str, dict[str, Any]] = {v: {} for v in variants}
    for v, eid, persona in results:
        by_variant[v][eid] = persona

    order = [eid for eid, _ in episodes]
    domains = {
        str(c.get("episode_id")): str(
            c.get("tau2_domain") or c.get("domain") or ""
        )
        for c in cases
        if c.get("episode_id")
    }
    from cue_training.evaluation.common.shuffle import build_shuffle_map, domain_of

    for eid in order:
        if not domains.get(eid):
            domains[eid] = domain_of(eid)
    shuffle_map = build_shuffle_map(order, domains=domains, seed=args.seed)

    rows: list[dict[str, Any]] = []
    for v in variants:
        personas = by_variant[v]
        diverse_texts: list[str] | None = None
        if v == "usp":
            try:
                from cue_training.baselines.usp.diverse import build_diverse_texts_from_lmsys

                diverse_texts = build_diverse_texts_from_lmsys(len(order), seed=args.seed)
            except Exception as exc:  # noqa: BLE001
                warn(
                    "tau2:personas",
                    f"USP LMSYS-USP diverse sampler failed ({exc}); skipping sample_diverse",
                )
        for i, eid in enumerate(order):
            own = personas[eid]
            own_val = _usp_persona_value(own) if v == "usp" else own
            origin_id = shuffle_map[eid]
            other = personas[origin_id]
            other_val = _usp_persona_value(other) if v == "usp" else other
            harness = _HARNESS_VARIANT.get(v, v)
            rows.append({
                "variant": f"baseline:{harness}", "arm": "paired", "episode_id": eid,
                "persona": own_val,
                "persona_origin_episode_id": eid,
                "shuffled_episode_id": origin_id,
            })
            rows.append({
                "variant": f"baseline:{harness}", "arm": "sample_shuffled", "episode_id": eid,
                "persona": other_val,
                "persona_origin_episode_id": origin_id,
                "shuffled_episode_id": origin_id,
            })
            if diverse_texts is not None:
                rows.append({
                    "variant": f"baseline:{harness}",
                    "arm": "sample_diverse",
                    "episode_id": eid,
                    "persona": diverse_texts[i],
                })

    write_jsonl(str(args.out), rows)
    log("tau2:personas", f"{len(episodes)} episodes x {variants} (native formats); "
        f"wrote {len(rows)} rows -> {args.out}")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Generate a tau2 persona sidecar (paired/sample_shuffled[/sample_diverse])."
    )
    p.add_argument("--cases", required=True, help="cases.paired.jsonl (episode_id + real_conversation)")
    p.add_argument("--out", required=True, help="Output personas sidecar JSONL.")
    p.add_argument("--variants", nargs="+", default=list(PERSONA_BASELINES),
                   help="Persona baselines to emit rows for (bare names or baseline:<name>).")
    p.add_argument("--sim_model", default="gpt-4o",
                   help="LLM for USP extract; RealUserSim always uses GPT-4o.")
    p.add_argument("--api_key_env", default="OPENAI_API_KEY")
    p.add_argument("--api_base", default=None)
    p.add_argument("--batch_size", type=int, default=50)
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--seed", type=int, default=0)
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    run(parse_args(argv))


if __name__ == "__main__":
    main()
