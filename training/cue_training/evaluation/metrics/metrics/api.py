from __future__ import annotations

import asyncio
import json
import os
import random
import threading
from statistics import mean
from typing import Any

from cue_training.evaluation.metrics import cache
from cue_training.evaluation.metrics.data import Episode, user_text
from cue_training.evaluation.metrics.stats import Aggregate, aggregate
from cue_training.runlog.log import warn

# Max in-flight LLM calls (async litellm). Override via env.
LLM_CONCURRENCY = int(os.environ.get("MIRROR_LLM_CONCURRENCY", "250"))


def _available_judges(judges: list[dict[str, str]]) -> list[dict[str, str]]:
    out: list[dict[str, str]] = []
    for j in judges:
        key = j.get("key") or ""
        # Local vLLM seats need no cloud key (LiteLLM accepts api_key="EMPTY").
        if key in ("", "HOSTED_VLLM_API_KEY") or os.environ.get(key):
            out.append(j)
    return out


def _is_hosted_vllm(model: str) -> bool:
    return (model or "").lower().startswith("hosted_vllm/")


def _turing_qwen_api_base() -> str | None:
    return (
        os.environ.get("MIRROR_TURING_QWEN_API_BASE")
        or os.environ.get("MIRROR_OSS_GEN_URL")
        or None
    )


def _turing_qwen_judge() -> dict[str, str]:
    """Qwen Turing seat: OpenRouter by default, or local vLLM via env.

    Local mode when ``MIRROR_TURING_QWEN_API_BASE`` is set (or the model is already
    ``hosted_vllm/...`` / ``Qwen/...``). On one 80GB GPU use ``Qwen/Qwen3-32B`` or
    ``Qwen/Qwen3-30B-A3B`` — the OpenRouter 2.4T MoE will not fit.
    """

    api_base = _turing_qwen_api_base()
    raw = os.environ.get("MIRROR_TURING_QWEN")
    use_local = bool(api_base) or (
        bool(raw) and (_is_hosted_vllm(raw) or raw.startswith("Qwen/"))
    )
    if use_local:
        hf_id = raw or "Qwen/Qwen3-32B"
        if hf_id.startswith("openrouter/"):
            hf_id = "Qwen/Qwen3-32B"
        model = hf_id if _is_hosted_vllm(hf_id) else f"hosted_vllm/{hf_id}"
        return {"name": "qwen-local", "model": model, "key": "HOSTED_VLLM_API_KEY"}
    return {
        "name": "qwen3.8",
        "model": raw or "openrouter/qwen/qwen3.8-2.4t-a95b",
        "key": "OPENROUTER_API_KEY",
    }


def _is_gemini3_plus(model: str) -> bool:
    """Gemini 3+ deprecates temperature/top_p/top_k sampling params (LiteLLM warning)."""

    m = (model or "").lower()
    return "gemini-3" in m or "/gemini-3" in m


def _is_gpt56_family(model: str) -> bool:
    m = (model or "").lower()
    return "gpt-5.6" in m


def _is_qwen_thinking_model(model: str) -> bool:
    m = (model or "").lower()
    return "qwen" in m and ("openrouter" in m or "qwen3" in m)


def _requires_temperature_one(model: str) -> bool:
    """Models that reject any temperature other than 1 (Claude Sonnet 5 / Fable / Opus 4.7+)."""

    m = (model or "").lower()
    return any(
        tag in m
        for tag in (
            "claude-sonnet-5",
            "fable",
            "opus-4-7",
            "opus_4_7",
            "opus-4.7",
            "opus_4.7",
            "opus-4-8",
            "opus_4_8",
            "opus-4.8",
            "opus_4.8",
        )
    )


def _completion_kwargs(model: str, prompt: str, *, max_tokens: int = 16) -> dict[str, Any]:
    # Gemini 3+: omit temperature/top_p/top_k (planned removal); put determinism in system text.
    if _is_gemini3_plus(model):
        messages = [
            {
                "role": "system",
                "content": (
                    "You are a deterministic evaluator. Follow the user instructions exactly. "
                    "Do not sample creatively; prefer the single most likely answer."
                ),
            },
            {"role": "user", "content": prompt},
        ]
        return {"model": model, "messages": messages, "max_tokens": max_tokens}
    # GPT-5.6 Terra: temperature fixed at 1; disable hidden reasoning for short A/B votes.
    if _is_gpt56_family(model):
        return {
            "model": model,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": max_tokens,
            "reasoning_effort": "none",
        }
    kwargs: dict[str, Any] = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 1 if _requires_temperature_one(model) else 0,
        "max_tokens": max_tokens,
    }
    # OpenRouter / hosted vLLM reject reasoning_effort.
    if "openrouter" not in (model or "").lower() and not _is_hosted_vllm(model):
        kwargs["reasoning_effort"] = "none"
    if _is_hosted_vllm(model):
        api_base = _turing_qwen_api_base() or "http://127.0.0.1:8000/v1"
        kwargs["api_base"] = api_base
        kwargs["api_key"] = os.environ.get("HOSTED_VLLM_API_KEY") or "EMPTY"
        kwargs["extra_body"] = {"chat_template_kwargs": {"enable_thinking": False}}
    elif _is_qwen_thinking_model(model):
        kwargs["chat_template_kwargs"] = {"enable_thinking": False}
    return kwargs


_LOOP_LOCK = threading.Lock()
_LOOP: asyncio.AbstractEventLoop | None = None


def _run_async(coro):
    """Run ``coro`` on one process-wide event loop and block for its result.

    litellm's global LoggingWorker binds its callback queue to whichever loop first
    ran a completion. A per-call ``asyncio.run`` from parallel scoring threads keeps
    rebinding that queue under the already-running worker, which then dies with
    "bound to a different event loop" / "task_done() called too many times" and leaks
    un-awaited success handlers. One shared loop keeps the binding stable.
    """

    global _LOOP
    with _LOOP_LOCK:
        if _LOOP is None or _LOOP.is_closed():
            _LOOP = asyncio.new_event_loop()
            threading.Thread(
                target=_LOOP.run_forever, name="metrics-llm-loop", daemon=True
            ).start()
        loop = _LOOP
    return asyncio.run_coroutine_threadsafe(coro, loop).result()


async def _acompletion_one(sem: asyncio.Semaphore, litellm, model: str, prompt: str, max_tokens: int) -> str | None:
    async with sem:
        try:
            resp = await litellm.acompletion(**_completion_kwargs(model, prompt, max_tokens=max_tokens))
            # A blank body is a failed call, not a vote — never let it score as a verdict.
            return (resp.choices[0].message.content or "").strip() or None
        except Exception as exc:  # noqa: BLE001
            warn("metrics", f"acompletion failed ({model}): {exc}")
            return None


async def _acompletion_gather(reqs: list[tuple[str, str, int]], concurrency: int) -> list[str | None]:
    import litellm

    sem = asyncio.Semaphore(concurrency)
    return await asyncio.gather(*(_acompletion_one(sem, litellm, m, p, mt) for (m, p, mt) in reqs))


def completion_batch(reqs: list[tuple[str, str, int]], *, concurrency: int | None = None) -> list[str | None]:
    """Run many chat completions concurrently via async litellm, bounded by ``concurrency``
    (default ``LLM_CONCURRENCY`` = 250). Returns a list aligned with ``reqs``; each item is the
    response text or None if that call failed. Each request is ``(model, prompt, max_tokens)``."""

    if not reqs:
        return []
    return _run_async(_acompletion_gather(reqs, concurrency or LLM_CONCURRENCY))


# --------------------------------------------------------------------------- naturalness Turing + paired audit

def _default_turing_judges() -> list[dict[str, str]]:
    # Claude-only for now: local Qwen showed strong A-position bias and OpenRouter
    # hit rate limits. Restore ``_turing_qwen_judge()`` when a reliable Qwen seat is ready.
    return [
        {
            "name": "claude-sonnet-5",
            "model": os.environ.get("MIRROR_TURING_CLAUDE", "anthropic/claude-sonnet-5"),
            "key": "ANTHROPIC_API_KEY",
        },
    ]


# Snapshot at import for tests / callers that read the module attribute; scoring
# rebuilds via ``_default_turing_judges()`` so env overrides apply at call time.
TURING_JUDGES: list[dict[str, str]] = _default_turing_judges()

TURING_EVAL_N = 60
TURING_FEWSHOT_PER_CLASS = 4
PAIRED_AUDIT_DIMENSIONS = (
    ("persona_affective", "Persona & Affective Traits: demeanor, emotional state, patience, personality cues"),
    ("linguistic_style", "Linguistic Style & Mechanics: vocabulary, phrasing, formality, typos, message-length patterns"),
    ("tech_competency", "Tech Competency & Knowledge: domain expertise, terminology, question depth"),
    ("interaction_flow", "Interaction & Data Flow: info-sharing habits (scattered vs dense), questioning style"),
    ("pacing_sequencing", "Pacing & Action Sequencing: turn length, topic transitions, conclusion patterns"),
)

_TERMINATION_PATTERNS = (
    "###STOP###",
    "###END###",
    "<END>",
    "</END>",
    "[END]",
    "END OF CONVERSATION",
    "end of conversation",
    "I am ending this conversation",
    "terminating the conversation",
)


def _strip_terminations(text: str) -> str:
    out = text or ""
    for pat in _TERMINATION_PATTERNS:
        out = out.replace(pat, " ")
    return " ".join(out.split())


def _hash_fewshot(examples: list[tuple[str, int]]) -> str:
    import hashlib

    payload = json.dumps(examples, ensure_ascii=False, sort_keys=True)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]


def _hash_pair(a: str, b: str) -> str:
    """Content hash so judgments are not reused across different proxy texts."""

    import hashlib

    payload = json.dumps([a, b], ensure_ascii=False)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]


def _parse_ab_tie(raw: str | None) -> str:
    if not raw:
        return "TIE"
    ans = raw.strip().upper()
    if "TIE" in ans:
        return "TIE"
    if ans.startswith("A"):
        return "A"
    if ans.startswith("B"):
        return "B"
    if " A" in f" {ans}" and "B" not in ans[:8]:
        return "A"
    if " B" in f" {ans}" and "A" not in ans[:8]:
        return "B"
    return "TIE"


def _domain_buckets(baseline: dict[str, Any]) -> dict[str, list[Any]]:
    buckets: dict[str, list[Any]] = {}
    for ep in baseline.values():
        meta = ep.metadata or {}
        raw = getattr(ep, "raw", None) or {}
        domain = str(
            meta.get("domain")
            or meta.get("tau2_domain")
            or raw.get("domain")
            or "all"
        )
        buckets.setdefault(domain, []).append(ep)
    return buckets


def _real_episode_id(ep: Any) -> str:
    """Aggregate fitting keys baselines as ``<episode_id>__<sim>``; scoring joins on the real id."""

    meta = getattr(ep, "metadata", None) or {}
    return str(meta.get("aggregate_source_episode_id") or ep.episode_id)


def _resolve_eval_episodes(id_list: list[str], episodes: list[Episode]) -> list[tuple[str, Episode]]:
    """Match fixed eval ids against candidates, tolerating stale ``<id>__<sim>`` state."""

    by_id = {ep.episode_id: ep for ep in episodes}
    out: list[tuple[str, Episode]] = []
    for eid in id_list:
        ep = by_id.get(eid)
        if ep is None and "__" in eid:
            ep = by_id.get(eid.rsplit("__", 1)[0])
        if ep is not None:
            out.append((eid, ep))
    return out


def fit_turing_state(
    baseline: dict[str, Any],
    cache_dir,
    *,
    n_eval: int = TURING_EVAL_N,
    n_fewshot: int = TURING_FEWSHOT_PER_CLASS,
    seed: int = 0,
) -> None:
    """Pick fixed eval episode ids + few-shot anchors from the baseline run only."""

    rng = random.Random(seed)
    eval_ids: dict[str, list[str]] = {}
    few_shot: dict[str, list[tuple[str, int]]] = {}
    for domain, eps in _domain_buckets(baseline).items():
        eligible = [ep for ep in eps if user_text(ep.human) and user_text(ep.base)]
        if not eligible:
            continue
        ids = [_real_episode_id(ep) for ep in eligible]
        rng.shuffle(ids)
        take = min(n_eval, max(0, len(ids) - 2 * n_fewshot))
        chosen = ids[:take] if take else ids[: min(n_eval, len(ids))]
        eval_ids[domain] = chosen
        pool = [ep for ep in eligible if _real_episode_id(ep) not in set(chosen)]
        if len(pool) < 2 * n_fewshot:
            pool = eligible
        rng.shuffle(pool)
        humans = [(user_text(ep.human), 1) for ep in pool[:n_fewshot]]
        bases = [(user_text(ep.base), 0) for ep in pool[n_fewshot : 2 * n_fewshot]]
        few_shot[domain] = humans + bases
    cache.save_pickle(cache_dir, "turing_eval_ids.pkl", eval_ids)
    cache.save_pickle(cache_dir, "turing_few_shot.pkl", few_shot)


def _turing_shots_block(examples: list[tuple[str, int]]) -> str:
    lines = []
    for text, label in examples:
        tag = "HUMAN" if label == 1 else "BASE_SIM"
        lines.append(f"[{tag}]\n{_strip_terminations(text)[:1200]}\n")
    return "\n".join(lines)


def _human_control_text(
    ep: Episode,
    episodes: list[Episode],
    baseline: dict[str, Any] | None,
) -> str | None:
    """Same-task different-user human when available, else same-domain other human."""

    task = str((ep.metadata or {}).get("task_id") or (ep.metadata or {}).get("tau2_task") or "")
    domain = ep.domain
    others = [o for o in episodes if o.episode_id != ep.episode_id and user_text(o.human)]
    if task:
        same_task = [
            o
            for o in others
            if str((o.metadata or {}).get("task_id") or (o.metadata or {}).get("tau2_task") or "") == task
        ]
        if same_task:
            return user_text(same_task[0].human)
    same_domain = [o for o in others if o.domain == domain]
    if same_domain:
        return user_text(same_domain[0].human)
    if baseline:
        for be in baseline.values():
            if be.episode_id != ep.episode_id and user_text(be.human):
                return user_text(be.human)
    return None


def turing_sonnet_qwen(
    episodes: list[Episode],
    cache_dir,
    *,
    baseline: dict[str, Any] | None = None,
    judges: list[dict[str, str]] | None = None,
    human_control: bool = False,
) -> Aggregate:
    """Naturalness Turing: corpus-level |0.5 - mean P(Human)| over judges × orderings.

    Votes are pooled across the eval subset first; the reported score is a single
    ``|0.5 - \\bar P|`` rather than the mean of per-episode ``|0.5 - P_e|``. A fair
    human–human control therefore lands near 0 even when the judge has a consistent
    (but balanced) preference on individual pairs.
    """

    panel = _available_judges(judges or _default_turing_judges())
    if not panel:
        return aggregate(
            "judge/turing_sonnet_qwen",
            [],
            {"skipped": "need ANTHROPIC_API_KEY"},
        )
    eval_ids = cache.load_pickle(cache_dir, "turing_eval_ids.pkl") or {}
    few_shot = cache.load_pickle(cache_dir, "turing_few_shot.pkl") or {}
    if not eval_ids:
        return aggregate("judge/turing_sonnet_qwen", [], {"skipped": "missing turing_eval_ids.pkl; refit baseline"})

    domain = episodes[0].domain if episodes else "all"
    id_list = list(eval_ids.get(domain) or eval_ids.get("all") or [])
    if not id_list and eval_ids:
        # Fall back to union when domain key mismatch (single-domain rollouts labeled oddly).
        id_list = sorted({i for ids in eval_ids.values() for i in ids})
    resolved = _resolve_eval_episodes(id_list, episodes)
    examples = list(few_shot.get(domain) or few_shot.get("all") or next(iter(few_shot.values()), []))
    shots = _turing_shots_block(examples)
    fewshot_hash = _hash_fewshot(examples)
    judgment_cache: dict[str, str] = cache.load_pickle(cache_dir, "turing_judgments.pkl") or {}

    reqs: list[tuple[str, str, int]] = []
    meta: list[tuple[str, str, str, bool]] = []  # episode_id, judge_name, order, cache_hit placeholder
    cache_hits: list[str | None] = []

    for eid, ep in resolved:
        human = _strip_terminations(user_text(ep.human))
        if human_control:
            other = _human_control_text(ep, episodes, baseline)
            proxy = _strip_terminations(other or "")
        else:
            proxy = _strip_terminations(user_text(ep.proxy))
        if not human or not proxy:
            continue
        pair_hash = _hash_pair(human, proxy)
        for order_name, a_is_human in (("H_first", True), ("P_first", False)):
            a, b = (human, proxy) if a_is_human else (proxy, human)
            prompt = (
                "You are judging which transcript is more likely written by a real human user "
                "(not a simulator). Use the labeled exemplars as calibration.\n\n"
                f"{shots}\n"
                "Now choose which unlabeled transcript is more human-like. Reply only A, B, or TIE.\n\n"
                f"A:\n{a[:4000]}\n\nB:\n{b[:4000]}"
            )
            for judge in panel:
                key = (
                    f"{judge['name']}|{eid}|{order_name}|{fewshot_hash}|"
                    f"{pair_hash}|{'hc' if human_control else 'sim'}"
                )
                cached = judgment_cache.get(key)
                if cached is not None:
                    cache_hits.append(cached)
                    meta.append((eid, judge["name"], order_name, a_is_human, pair_hash))
                    continue
                cache_hits.append(None)
                meta.append((eid, judge["name"], order_name, a_is_human, pair_hash))
                reqs.append((judge["model"], prompt, 16))

    fresh = completion_batch(reqs) if reqs else []
    fi = 0
    answers: list[str | None] = []
    for cached in cache_hits:
        if cached is not None:
            answers.append(cached)
        else:
            ans = fresh[fi] if fi < len(fresh) else None
            fi += 1
            answers.append(ans)

    # Persist new judgments.
    for (eid, jname, order_name, _a_is_h, pair_hash), ans in zip(meta, answers):
        if not (ans or "").strip():
            continue
        key = (
            f"{jname}|{eid}|{order_name}|{fewshot_hash}|"
            f"{pair_hash}|{'hc' if human_control else 'sim'}"
        )
        judgment_cache[key] = ans
    cache.save_pickle(cache_dir, "turing_judgments.pkl", judgment_cache)

    per_ep: dict[str, list[float]] = {}
    n_failed = 0
    for (eid, _jname, _order, a_is_human, _ph), ans in zip(meta, answers):
        if not (ans or "").strip():
            n_failed += 1
            continue
        vote = _parse_ab_tie(ans)
        if vote == "TIE":
            p_human = 0.5
        elif vote == "A":
            p_human = 1.0 if a_is_human else 0.0
        else:
            p_human = 0.0 if a_is_human else 1.0
        per_ep.setdefault(eid, []).append(p_human)

    # Corpus-level: pool episode means, then one |0.5 - P̄|.
    p_means = [mean(vals) for vals in per_ep.values() if vals]
    mean_p = mean(p_means) if p_means else None
    scores = [abs(0.5 - mean_p)] if mean_p is not None else []
    extras = {
        "judges": [j["name"] for j in panel],
        "n_eval_ids": len(id_list),
        "n_matched_episodes": len(resolved),
        "n_failed_calls": n_failed,
        "n_scored": len(p_means),
        "n_votes": sum(len(v) for v in per_ep.values()),
        "fewshot_hash": fewshot_hash,
        "human_control": human_control,
        "mean_p_human": mean_p,
        "aggregation": "corpus_|0.5-mean_P|",
        "unit": "|0.5-mean_P(human)|",
        "lower_is_better": True,
    }
    if not scores:
        extras["skipped"] = (
            f"no eval episode ids matched candidates (n_eval_ids={len(id_list)})"
            if not resolved
            else f"all {n_failed} judge calls failed"
        )
    return aggregate("judge/turing_sonnet_qwen", scores, extras)


def paired_audit_fidelity(
    episodes: list[Episode],
    cache_dir,
    *,
    judges: list[dict[str, str]] | None = None,
    human_control: bool = False,
) -> Aggregate:
    """RealUserSim Paired Audit / Fidelity Index over five conversational-identity dimensions."""

    panel = _available_judges(judges or _default_turing_judges())
    if not panel:
        return aggregate(
            "mimicry/paired_audit",
            [],
            {"skipped": "need ANTHROPIC_API_KEY"},
        )
    eval_ids = cache.load_pickle(cache_dir, "turing_eval_ids.pkl") or {}
    domain = episodes[0].domain if episodes else "all"
    id_list = list(eval_ids.get(domain) or eval_ids.get("all") or [])
    if not id_list and eval_ids:
        id_list = sorted({i for ids in eval_ids.values() for i in ids})
    resolved = _resolve_eval_episodes(id_list, episodes)
    dim_lines = "\n".join(f"- {key}: {desc}" for key, desc in PAIRED_AUDIT_DIMENSIONS)
    judgment_cache: dict[str, str] = cache.load_pickle(cache_dir, "paired_audit_judgments.pkl") or {}

    reqs: list[tuple[str, str, int]] = []
    meta: list[tuple[str, str, str, str]] = []
    cache_hits: list[str | None] = []

    for eid, ep in resolved:
        th = _strip_terminations(user_text(ep.human))
        if human_control:
            ts = _strip_terminations(_human_control_text(ep, episodes, None) or "")
        else:
            ts = _strip_terminations(user_text(ep.proxy))
        if not th or not ts:
            continue
        pair_hash = _hash_pair(th, ts)
        for order_name, th_first in (("Th_first", True), ("Ts_first", False)):
            a, b = (th, ts) if th_first else (ts, th)
            prompt = (
                "Compare two unlabeled user trajectories from the same task scenario. "
                "For EACH dimension below, reply MATCH or NO_MATCH (whether A and B reflect "
                "the same conversational identity).\n"
                f"{dim_lines}\n\n"
                "Reply with one line per dimension as: <key>: MATCH|NO_MATCH\n\n"
                f"Trajectory A:\n{a[:3500]}\n\nTrajectory B:\n{b[:3500]}"
            )
            for judge in panel:
                key = f"{judge['name']}|{eid}|{order_name}|{pair_hash}|{'hc' if human_control else 'sim'}"
                cached = judgment_cache.get(key)
                cache_hits.append(cached)
                meta.append((eid, judge["name"], order_name, pair_hash))
                if cached is None:
                    reqs.append((judge["model"], prompt, 256))

    fresh = completion_batch(reqs) if reqs else []
    fi = 0
    answers: list[str | None] = []
    for cached in cache_hits:
        if cached is not None:
            answers.append(cached)
        else:
            ans = fresh[fi] if fi < len(fresh) else None
            fi += 1
            answers.append(ans)

    for (eid, jname, order_name, pair_hash), ans in zip(meta, answers):
        if not (ans or "").strip():
            continue
        key = f"{jname}|{eid}|{order_name}|{pair_hash}|{'hc' if human_control else 'sim'}"
        judgment_cache[key] = ans
    cache.save_pickle(cache_dir, "paired_audit_judgments.pkl", judgment_cache)

    def _parse_dims(raw: str | None) -> dict[str, float] | None:
        """Parsed verdicts, or None when the reply names no dimension (blank/garbled)."""

        if not (raw or "").strip():
            return None
        out = {k: 0.0 for k, _ in PAIRED_AUDIT_DIMENSIONS}
        upper = raw.upper()
        found = False
        for key, _desc in PAIRED_AUDIT_DIMENSIONS:
            # Look for KEY: MATCH / NO_MATCH near the key name.
            idx = upper.find(key.upper())
            if idx < 0:
                continue
            found = True
            snippet = upper[idx : idx + 40]
            if "NO_MATCH" in snippet or "NO-MATCH" in snippet or "NO MATCH" in snippet:
                out[key] = 0.0
            elif "MATCH" in snippet:
                out[key] = 1.0
        return out if found else None

    per_ep_scores: dict[str, list[float]] = {}
    dim_rates: dict[str, list[float]] = {k: [] for k, _ in PAIRED_AUDIT_DIMENSIONS}
    per_judge: dict[str, list[float]] = {j["name"]: [] for j in panel}
    n_failed = 0
    for (eid, jname, _order, _ph), ans in zip(meta, answers):
        dims = _parse_dims(ans)
        if dims is None:
            n_failed += 1
            continue
        fidelity = mean(dims.values())
        per_ep_scores.setdefault(eid, []).append(fidelity)
        per_judge[jname].append(fidelity)
        for k, v in dims.items():
            dim_rates[k].append(v)

    scores = [mean(v) for v in per_ep_scores.values() if v]
    extras = {
        "judges": [j["name"] for j in panel],
        "per_judge": {name: (mean(v) if v else None) for name, v in per_judge.items()},
        "per_dimension": {k: (mean(v) if v else None) for k, v in dim_rates.items()},
        "n_eval_ids": len(id_list),
        "n_matched_episodes": len(resolved),
        "n_failed_calls": n_failed,
        "n_scored": len(scores),
        "human_control": human_control,
        "unit": "fidelity_index",
        "higher_is_better": True,
    }
    if not scores:
        extras["skipped"] = (
            f"no eval episode ids matched candidates (n_eval_ids={len(id_list)})"
            if not resolved
            else f"all {n_failed} judge calls failed"
        )
    return aggregate("mimicry/paired_audit", scores, extras)

