"""Aggregate metrics: fit once on floor-base sims, score every method, pool seeds + domains."""

from __future__ import annotations

import argparse
import json
import math
import os
import re
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from cue_training.utils.config import storage_root
from cue_training.evaluation.metrics.data import (
    _HUMAN_KEYS,
    _ROLLOUT_KEYS,
    _first_turns,
    _metadata,
    episode_key,
    read_jsonl,
)
from cue_training.evaluation.metrics.display import _cell
from cue_training.evaluation.metrics.run import (
    DEFAULT_METRICS,
    _parse_metrics,
    fit_aggregate_state,
    run_one,
)
from cue_training.evaluation.metrics.stats import Aggregate, aggregate_from_json, aggregate_seed_means
from cue_training.runlog.log import log, warn

TAG = "metrics.aggregate"

_ROLLOUT_GLOBS = (
    "rollout.tau2.jsonl",
    "rollout.simulatorarena.jsonl",
    "rollout.prism.jsonl",
)
_HUMAN_LABEL = "human"
_SUCCESS_THRESHOLD = 1.0
_SEED_DIR_RE = re.compile(r"^seed-(\d+)$")
_SEED_SUFFIX_RE = re.compile(r"[-_]seed[-_]?(\d+)$", re.IGNORECASE)
# Benchmarks that get an equal-weight macro-average across domains (plus seed CIs).
_MACRO_DOMAIN_BENCHMARKS = frozenset({"tau2", "simulatorarena"})
# tau2 Success F1 stores raw env success under extras; promote it to its own column.
_SUCCESS_F1_METRIC = "env/tau2_task_success"
_SUCCESS_RATE_METRIC = "env/tau2_success_rate"


@dataclass(frozen=True)
class CandidateJob:
    """One complete rollout job discovered under the rollouts root."""

    run_id: str
    benchmark: str
    domain: str
    kind: str  # baseline | cue | human
    method: str
    simulator: str
    rollout_path: Path
    seed: int | None = None
    family: str = ""

    @property
    def label(self) -> str:
        # Prefer run_id so cue-general vs cue-general_final stay distinct before seed-stripping.
        return self.run_id


def method_family(run_id: str, *, kind: str, method: str) -> str:
    """Canonical method key shared across seeds (strip ``-seed-N`` / ``_seed_N``)."""

    if kind == _HUMAN_LABEL or run_id == _HUMAN_LABEL:
        return _HUMAN_LABEL
    stripped = _SEED_SUFFIX_RE.sub("", run_id).rstrip("-_")
    # Multi-method / multiseed baseline runs share one run_id; distinguish by method
    # folder — including ``base`` so per-seed base pools as "base" (not the run_id).
    if kind == "baseline" and method and (
        stripped in {"baseline", "base"}
        or stripped.startswith("baseline")
        or "multiseed" in stripped
    ):
        return method
    if kind == "cue" and stripped:
        return stripped
    if method and method not in {"general", "proposals", "refined"}:
        # Prefer concrete baseline method names when run_id is generic.
        if stripped in {"baseline", "base"} or not stripped:
            return method
    return stripped or method or run_id


def write_human_proxy_rollout(
    baseline_path: Path,
    out_path: Path,
    *,
    tau_usi_source: str | None = None,
) -> Path:
    """Write a candidate JSONL where proxy == human, for the aggregate upper-bound row."""

    rewards: dict[str, float] = {}
    try:
        from cue_training.evaluation.metrics.metrics import tau_usi

        rewards = tau_usi.human_rewards(tau_usi_source)
    except Exception as exc:  # noqa: BLE001
        warn(TAG, f"human upper-bound: tau-usi rewards unavailable ({exc})")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with out_path.open("w", encoding="utf-8") as handle:
        for record in read_jsonl(baseline_path):
            key = episode_key(record)
            human = _first_turns(record, _HUMAN_KEYS)
            if not key or not human:
                continue
            meta = _metadata(record)
            meta["arm"] = _HUMAN_LABEL
            meta["upper_bound"] = True
            reward = rewards.get(key)
            if reward is None:
                reward = rewards.get(str(meta.get("instance_id") or ""))
            if isinstance(reward, (int, float)):
                meta["reward"] = float(reward)
                meta["is_success"] = float(reward) >= _SUCCESS_THRESHOLD - 1e-6
            row = {
                **{k: v for k, v in record.items() if k not in ("metadata",)},
                "episode_id": key,
                "real_conversation": human,
                "rollout_conversation": human,
                "metadata": meta,
            }
            for key_name in _ROLLOUT_KEYS:
                if key_name != "rollout_conversation":
                    row.pop(key_name, None)
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
            n += 1
    if n == 0:
        raise ValueError(f"no human episodes writable from {baseline_path}")
    log(TAG, f"human upper-bound candidate -> {out_path} ({n} episodes)")
    return out_path


def _find_rollout_jsonl(job_dir: Path) -> Path | None:
    for name in _ROLLOUT_GLOBS:
        path = job_dir / name
        if path.is_file():
            return path
    matches = sorted(job_dir.glob("rollout.*.jsonl"))
    return matches[0] if matches else None


def seed_roots(rollouts_root: Path, seeds: list[int] | None) -> list[tuple[int | None, Path]]:
    """Resolve ``(seed, root)`` search roots under a rollouts tree.

    Prefer ``rollouts_root/seed-{N}/`` when those dirs exist; otherwise use the flat root
    once (seed=None). Explicit ``seeds`` filters which seed dirs to include.
    """

    root = Path(rollouts_root)
    found: list[tuple[int, Path]] = []
    for child in sorted(root.iterdir()) if root.is_dir() else []:
        match = _SEED_DIR_RE.match(child.name)
        if match and child.is_dir():
            found.append((int(match.group(1)), child))
    if found:
        if seeds is None:
            return [(s, p) for s, p in found]
        want = set(int(s) for s in seeds)
        return [(s, p) for s, p in found if s in want]
    return [(None, root)]


def _base_paths_under_run(
    seed_root: Path,
    run_id: str,
    *,
    benchmark: str,
    domain: str,
    simulators: list[str],
) -> list[tuple[str, Path]]:
    """Collect floor-base simulator rollouts under one run-id folder."""

    out: list[tuple[str, Path]] = []
    for sim in simulators:
        job_dir = seed_root / run_id / benchmark / domain / "baseline" / "base" / sim
        path = _find_rollout_jsonl(job_dir)
        if path is not None:
            out.append((sim, path))
    return out


def _candidate_base_run_ids(seed_root: Path, base_run_id: str, seed: int | None) -> list[str]:
    """Ordered run-id guesses for floor-base: exact, seeded suffix, then on-disk fallbacks."""

    ordered: list[str] = [base_run_id]
    if seed is not None:
        ordered.extend(
            (
                f"{base_run_id}-seed-{seed}",
                f"{base_run_id}_seed_{seed}",
            )
        )
    if seed_root.is_dir():
        extras = sorted(
            (
                child.name
                for child in seed_root.iterdir()
                if child.is_dir()
                and child.name not in ordered
                and (
                    base_run_id in child.name
                    or "multiseed" in child.name
                    or child.name.startswith("baseline")
                    or child.name == "base"
                )
            ),
            key=lambda name: (
                0 if base_run_id in name else 1,
                0 if "multiseed" in name else 1,
                0 if name.startswith("baseline") else 1,
                name,
            ),
        )
        ordered.extend(extras)
    return ordered


def resolve_base_paths(
    rollouts_root: Path,
    *,
    benchmark: str,
    domain: str,
    base_run_id: str = "base",
    simulators: list[str],
    seeds: list[int] | None = None,
) -> list[tuple[str, Path]]:
    """Return (sim, rollout_path) for each floor-base simulator that exists on disk.

    Accepts a dedicated ``base-once`` run-id or base nested in a seeded / multiseed
    baseline run (e.g. ``hosted_baselines_multiseed-seed-0``).
    """

    # Prefer the first seed root that yields >=2 sims (usually seed-0).
    for seed, seed_root in seed_roots(rollouts_root, seeds):
        last_partial: list[tuple[str, Path]] = []
        last_run = base_run_id
        for run_id in _candidate_base_run_ids(seed_root, base_run_id, seed):
            out = _base_paths_under_run(
                seed_root,
                run_id,
                benchmark=benchmark,
                domain=domain,
                simulators=simulators,
            )
            if len(out) >= 2:
                if run_id != base_run_id:
                    log(
                        TAG,
                        f"floor-base resolved via run_id={run_id} "
                        f"(requested base_run_id={base_run_id})",
                    )
                return out
            if out:
                last_partial, last_run = out, run_id
        if last_partial:
            warn(
                TAG,
                f"only {len(last_partial)} floor-base sim(s) under "
                f"{seed_root / last_run}; need >=2",
            )
    # Flat fallback (no seed-* dirs, or bases live at rollouts_root directly).
    root = Path(rollouts_root)
    for run_id in _candidate_base_run_ids(root, base_run_id, None):
        out = _base_paths_under_run(
            root,
            run_id,
            benchmark=benchmark,
            domain=domain,
            simulators=simulators,
        )
        if out:
            return out
    return []


def discover_candidates(
    rollouts_root: Path,
    *,
    benchmark: str,
    domain: str,
    methods: set[str] | None = None,
    seeds: list[int] | None = None,
) -> list[CandidateJob]:
    """Walk seed/flat rollouts trees for ``COMPLETE`` jobs under ``{benchmark}/{domain}``."""

    jobs: list[CandidateJob] = []
    seen: set[tuple[Any, ...]] = set()
    for seed, seed_root in seed_roots(rollouts_root, seeds):
        pattern = f"*/{benchmark}/{domain}/*/*/*/COMPLETE"
        for complete in sorted(seed_root.glob(pattern)):
            job_dir = complete.parent
            try:
                sim = job_dir.name
                method = job_dir.parent.name
                kind = job_dir.parent.parent.name
                domain_part = job_dir.parent.parent.parent.name
                bench_part = job_dir.parent.parent.parent.parent.name
                run_id = job_dir.parent.parent.parent.parent.parent.name
            except Exception:  # noqa: BLE001
                continue
            if domain_part != domain or bench_part != benchmark:
                continue
            if kind not in ("baseline", "cue"):
                continue
            family = method_family(run_id, kind=kind, method=method)
            if methods is not None and run_id not in methods and method not in methods and family not in methods:
                continue
            rollout = _find_rollout_jsonl(job_dir)
            if rollout is None:
                warn(TAG, f"COMPLETE but no rollout JSONL under {job_dir}")
                continue
            key = (seed, run_id, domain, kind, method, sim)
            if key in seen:
                continue
            seen.add(key)
            jobs.append(
                CandidateJob(
                    run_id=run_id,
                    benchmark=benchmark,
                    domain=domain,
                    kind=kind,
                    method=method,
                    simulator=sim,
                    rollout_path=rollout,
                    seed=seed,
                    family=family,
                )
            )
    return jobs


def _fmt_cell(agg: Aggregate | None) -> str:
    if agg is None:
        return "—"
    return _cell(agg)


def leaderboard_metric_names(metric_names: list[str]) -> list[str]:
    """Insert Success rate immediately before Success F1 when the latter is scored."""

    out: list[str] = []
    for name in metric_names:
        if name == _SUCCESS_F1_METRIC:
            out.append(_SUCCESS_RATE_METRIC)
        out.append(name)
    return out


def metrics_present_in_summaries(summaries: list[dict[str, Any]]) -> list[str]:
    """Ordered metric names found in summaries (DEFAULT_METRICS first, then extras)."""

    seen: set[str] = set()
    for summary in summaries:
        by_group = summary.get("aggregates_by_group") or summary.get("aggregates_by_arm") or {}
        for aggs in by_group.values():
            for raw in aggs or []:
                name = (
                    raw.get("metric_name")
                    if isinstance(raw, dict)
                    else getattr(raw, "metric_name", None)
                )
                if name:
                    seen.add(str(name))
    ordered = [m for m in DEFAULT_METRICS if m in seen]
    ordered.extend(m for m in sorted(seen) if m not in ordered)
    return ordered


def _env_success_rate_payload(env_agg: Aggregate | None) -> dict[str, Any] | None:
    """Lift ``env_success_rate`` extras into a synthetic Aggregate-shaped dict."""

    if env_agg is None:
        return None
    esr = (env_agg.extras or {}).get("env_success_rate") or {}
    mean = esr.get("mean")
    if mean is None:
        return None
    try:
        mean_f = float(mean)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(mean_f):
        return None
    return {
        "metric_name": _SUCCESS_RATE_METRIC,
        "mean": mean_f,
        "standard_deviation": esr.get("standard_deviation") or esr.get("stdev"),
        "confidence_interval": esr.get("ci") if esr.get("ci") is not None else esr.get("confidence_interval"),
        "sample_size": int(esr.get("n") or esr.get("sample_size") or 0),
        "extras": {},
    }


def _metric_mean(raw: Any) -> float | None:
    if raw is None:
        return None
    if isinstance(raw, Aggregate):
        if raw.extras.get("skipped") or raw.extras.get("error"):
            return None
        val = raw.mean
    elif isinstance(raw, dict):
        if (raw.get("extras") or {}).get("skipped") or (raw.get("extras") or {}).get("error"):
            return None
        val = raw.get("mean")
    else:
        return None
    try:
        f = float(val)
    except (TypeError, ValueError):
        return None
    return f if math.isfinite(f) else None


def summaries_to_rows(
    summaries: list[dict[str, Any]],
    *,
    metric_names: list[str],
) -> list[dict[str, Any]]:
    """Flatten per-candidate summaries into leaderboard-style rows (one seed each)."""

    rows: list[dict[str, Any]] = []
    for summary in summaries:
        meta = summary.get("candidate_meta") or {}
        by_group = summary.get("aggregates_by_group") or summary.get("aggregates_by_arm") or {}
        group_meta = summary.get("group_meta") or {}
        for group, aggs_raw in by_group.items():
            aggs = [
                aggregate_from_json(a) if isinstance(a, dict) else a
                for a in (aggs_raw or [])
                if isinstance(a, (dict, Aggregate))
            ]
            by_name = {a.metric_name: a for a in aggs}
            gmeta = group_meta.get(group) or {}
            arm = str(gmeta.get("arm") or str(group).split("/")[0])
            if "/" in str(group):
                domain = str(gmeta.get("domain") or str(group).split("/", 1)[1] or "all")
            else:
                # Arm-only key: trust group_meta, but never fall back to candidate_meta
                # (umbrella folders like customer-service / a sibling subdomain stamp).
                domain = str(gmeta.get("domain") or "all")
            family = str(
                meta.get("family")
                or method_family(
                    str(meta.get("run_id") or ""),
                    kind=str(meta.get("kind") or ""),
                    method=str(meta.get("method") or ""),
                )
            )
            success_rate = _env_success_rate_payload(by_name.get(_SUCCESS_F1_METRIC))
            metrics_payload: dict[str, Any] = {}
            seed_means: dict[str, float | None] = {}
            for name in metric_names:
                if name == _SUCCESS_RATE_METRIC:
                    metrics_payload[name] = success_rate
                    seed_means[name] = _metric_mean(success_rate)
                elif name in by_name:
                    metrics_payload[name] = by_name[name].to_json()
                    seed_means[name] = _metric_mean(by_name[name])
                else:
                    metrics_payload[name] = None
                    seed_means[name] = None
            row: dict[str, Any] = {
                "run_id": meta.get("run_id"),
                "family": family,
                "method": meta.get("method"),
                "kind": meta.get("kind"),
                "simulator": meta.get("simulator"),
                "seed": meta.get("seed"),
                "arm": arm,
                "domain": domain,
                "group": group,
                "n": next((a.sample_size for a in aggs if a.sample_size), 0),
                "metrics": metrics_payload,
                "seed_means": seed_means,
            }
            rows.append(row)
    return rows


def pool_across_seeds(
    rows: list[dict[str, Any]],
    *,
    metric_names: list[str],
) -> list[dict[str, Any]]:
    """Average metric means across seeds for each (family, simulator, arm, domain).

    ± on the leaderboard is the sample standard deviation of the per-seed means.
    """

    buckets: dict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        family = row.get("family") or row.get("run_id")
        key = (family, row.get("simulator"), row.get("arm"), row.get("domain"), row.get("kind"))
        buckets[key].append(row)

    pooled: list[dict[str, Any]] = []
    for (family, sim, arm, domain, kind), group_rows in sorted(buckets.items(), key=lambda kv: str(kv[0])):
        metrics_out: dict[str, Any] = {}
        seed_values: dict[str, list[float]] = {}
        for name in metric_names:
            vals = []
            for r in group_rows:
                means = r.get("seed_means") or {}
                v = means.get(name)
                if v is None:
                    v = _metric_mean((r.get("metrics") or {}).get(name))
                if v is not None:
                    vals.append(float(v))
            seed_values[name] = vals
            if not vals:
                metrics_out[name] = None
            else:
                agg = aggregate_seed_means(
                    name, vals, extras={"n_seeds": len(vals), "seed_means": vals}
                )
                metrics_out[name] = agg.to_json()
        seeds = sorted({r.get("seed") for r in group_rows if r.get("seed") is not None})
        pooled.append({
            "run_id": family,
            "family": family,
            "method": group_rows[0].get("method"),
            "kind": kind,
            "simulator": sim,
            "arm": arm,
            "domain": domain,
            "group": f"{arm}/{domain}" if domain and domain != "all" else arm,
            "n_seeds": len(seeds) if seeds else len(group_rows),
            "seeds": seeds,
            "n": max((int(r.get("n") or 0) for r in group_rows), default=0),
            "metrics": metrics_out,
            "pooled": "seeds",
        })
    return pooled


def macro_across_domains(
    seed_rows: list[dict[str, Any]],
    *,
    metric_names: list[str],
    domains: list[str],
) -> list[dict[str, Any]]:
    """Equal-weight mean across domains, with ± = stdev of per-seed macros.

    For each seed: macro = mean of that seed's per-domain scores (domains equally weighted).
    Across seeds: report mean ± sample stdev of those macros.
    """

    folder_domains = {str(d) for d in domains}
    # When a single umbrella folder is scored (e.g. tau2 customer-service), group-level
    # domains (airline/retail/...) are the ones to average — include every non-empty domain.
    umbrella = len(folder_domains) == 1 and next(iter(folder_domains)) not in {"writing", "math"}

    by_seed: dict[tuple[Any, ...], dict[str, dict[str, float]]] = defaultdict(dict)
    meta_by_key: dict[tuple[Any, ...], dict[str, Any]] = {}
    for row in seed_rows:
        domain = str(row.get("domain") or "")
        # Skip arm-pooled / placeholder domains — only real subdomains enter the macro.
        if not domain or domain in {"macro", "all", "pooled"}:
            continue
        if not umbrella and folder_domains and domain not in folder_domains:
            continue
        family = row.get("family") or row.get("run_id")
        key = (family, row.get("simulator"), row.get("arm"), row.get("kind"), row.get("seed"))
        meta_by_key[(family, row.get("simulator"), row.get("arm"), row.get("kind"))] = row
        means = row.get("seed_means") or {}
        slot: dict[str, float] = {}
        for name in metric_names:
            v = means.get(name)
            if v is None:
                v = _metric_mean((row.get("metrics") or {}).get(name))
            if v is not None:
                slot[name] = float(v)
        if not slot:
            continue
        # Merge into the domain slot instead of replacing — a later partial row (e.g. arm-pooled
        # MAUVE wrongly stamped as a subdomain) must not erase metrics already recorded.
        existing = by_seed[key].setdefault(domain, {})
        for name, val in slot.items():
            existing.setdefault(name, val)

    families: dict[tuple[Any, ...], list[tuple[Any, dict[str, dict[str, float]]]]] = defaultdict(list)
    for (family, sim, arm, kind, seed), domain_map in by_seed.items():
        families[(family, sim, arm, kind)].append((seed, domain_map))

    out: list[dict[str, Any]] = []
    for (family, sim, arm, kind), seed_items in sorted(families.items(), key=lambda kv: str(kv[0])):
        all_domains = sorted({d for _, dmap in seed_items for d in dmap})
        if len(all_domains) < 2:
            continue
        metrics_out: dict[str, Any] = {}
        for name in metric_names:
            macros: list[float] = []
            for _seed, dmap in seed_items:
                vals = [dmap[d][name] for d in all_domains if d in dmap and name in dmap[d]]
                if not vals:
                    continue
                macros.append(sum(vals) / len(vals))
            if not macros:
                metrics_out[name] = None
            else:
                agg = aggregate_seed_means(
                    name,
                    macros,
                    extras={
                        "n_seeds": len(macros),
                        "seed_macros": macros,
                        "domains": all_domains,
                        "pooled": "seeds_of_domain_macro",
                    },
                )
                metrics_out[name] = agg.to_json()
        seeds = sorted({s for s, _ in seed_items if s is not None})
        out.append({
            "run_id": family,
            "family": family,
            "method": (meta_by_key.get((family, sim, arm, kind)) or {}).get("method"),
            "kind": kind,
            "simulator": sim,
            "arm": arm,
            "domain": "macro",
            "group": f"{arm}/macro",
            "n_seeds": len(seeds) if seeds else len(seed_items),
            "seeds": seeds,
            "domains_averaged": all_domains,
            "n": 0,
            "metrics": metrics_out,
            "pooled": "domain_macro",
        })
    return out


def build_leaderboard(
    rows: list[dict[str, Any]],
    *,
    metric_names: list[str],
    title_note: str = "",
) -> tuple[list[dict[str, Any]], str]:
    """Render rows into markdown; rows are already pooled (or single-seed)."""

    ordered = sorted(
        rows,
        key=lambda r: (
            0 if r.get("run_id") == _HUMAN_LABEL or r.get("kind") == _HUMAN_LABEL else 1,
            0 if r.get("domain") != "macro" else 1,
            str(r.get("family") or r.get("run_id") or ""),
            str(r.get("simulator") or ""),
            str(r.get("arm") or ""),
            str(r.get("domain") or ""),
        ),
    )

    # Unlisted names fall through to themselves, so historical runs still render.
    headers = {
        "env/tau2_success_rate": "Success rate",
        "env/tau2_task_success": "Success F1",
        "classifier/sim2real": "Sim2Real P(human)",
        "judge/turing_sonnet_qwen": "Turing |0.5-mean_P|",
        "mimicry/wegmann_ava": "Wegmann AVA",
        "mimicry/paired_audit": "Paired audit",
        "coverage/styledistance_behavioral": "SD coverage",
        "coverage/sim2real_behavioral": "S2R Chamfer",
    }
    note = title_note or (
        "Scores are means across seeds with ± sample standard deviation of the per-seed means. "
        "`domain=macro` is an equal-weight mean of per-domain scores (taken per seed first); "
        "its ± is the sample stdev of those per-seed macros."
    )
    lines = [
        "# Aggregate metrics leaderboard",
        "",
        "Fitted once on all floor-`base` simulators + human data; every method/sim scored "
        "against that shared state. The `human` row is an upper bound (proxy = real user turns).",
        "",
        note,
        "",
    ]
    sections = (
        ("Task", [name for name in metric_names if name.startswith("env/")]),
        (
            "Naturalness",
            [
                name
                for name in metric_names
                if name.startswith("classifier/") or name.startswith("judge/")
            ],
        ),
        (
            "Mimicry",
            [
                name
                for name in metric_names
                if name.startswith("mimicry/") or name.startswith("authenticity/")
            ],
        ),
        ("Coverage", [name for name in metric_names if name.startswith("coverage/")]),
    )
    assigned = {name for _title, names in sections for name in names}
    other = [name for name in metric_names if name not in assigned]
    rendered_sections = [*sections, *((("Other", other),) if other else ())]
    for title, names in rendered_sections:
        if not names:
            continue
        cols = [headers.get(name, name) for name in names]
        lines.extend(
            [
                f"## {title}",
                "",
                "| Method | Simulator | Arm | Domain | Seeds | "
                + " | ".join(cols)
                + " |",
                "| --- | --- | --- | --- | --- | "
                + " | ".join(["---"] * len(cols))
                + " |",
            ]
        )
        for row in ordered:
            cells = []
            for name in names:
                raw = (row.get("metrics") or {}).get(name)
                agg = aggregate_from_json(raw) if isinstance(raw, dict) else None
                cells.append(_fmt_cell(agg))
            method = row.get("family") or row.get("run_id") or row.get("method") or "?"
            n_seeds = row.get("n_seeds") or (len(row.get("seeds") or []) or "—")
            lines.append(
                f"| {method} | {row.get('simulator')} | {row.get('arm')} | "
                f"{row.get('domain')} | {n_seeds} | "
                + " | ".join(cells)
                + " |"
            )
        lines.append("")
    return ordered, "\n".join(lines)


def _score_job(
    *,
    job: CandidateJob,
    join_baseline: Path,
    base_paths: list[Path],
    runs_dir: Path,
    metric_names: list[str],
    oss_model: str,
    tau_usi_source: str | None,
    state_dir: Path,
    fma_run: Path | None = None,
    luar_model: str = "rrivera1849/LUAR-MUD",
    cue_checkpoint: str | None = None,
    cue_session_preprocess: str = "auto",
    merge: bool = False,
) -> dict[str, Any] | None:
    from cue_training.evaluation.metrics.fma_env import env_error_ids_for_job

    seed_part = f"seed-{job.seed}" if job.seed is not None else "seed-na"
    run_dir = runs_dir / seed_part / job.family / job.simulator
    fma_dir = None
    if fma_run is not None:
        fma_dir = Path(fma_run).expanduser()
        if not (fma_dir.is_dir() or fma_dir.is_absolute() or "/" in str(fma_run) or str(fma_run).startswith(".")):
            from cue_training.utils.config import storage_root as _storage_root

            fma_dir = (_storage_root() / "fma" / str(fma_run)).resolve()
        else:
            fma_dir = fma_dir.resolve()
    # Default arm for FMA join; also union any arm under this method/sim prefix.
    env_ids = env_error_ids_for_job(
        fma_dir,
        kind=job.kind,
        method=job.method,
        simulator=job.simulator,
        arm="as_is",
    )
    if fma_dir is not None:
        from cue_training.evaluation.common.io import read_jsonl
        from cue_training.evaluation.metrics.fma_env import ENVIRONMENT_ERROR_LABEL, tagged_path

        tagged = tagged_path(fma_dir)
        if tagged.is_file():
            prefix = (
                f"cue-{job.method}-{job.simulator}|"
                if job.kind == "cue" or job.method in {"general", "refined", "proposals"}
                else (
                    f"base-{job.simulator}|"
                    if job.method == "base"
                    else f"{job.method}-{job.simulator}|"
                )
            )
            for row in read_jsonl(str(tagged)):
                if str(row.get("label") or "").strip() != ENVIRONMENT_ERROR_LABEL:
                    continue
                sid = str(row.get("source_id") or "")
                if sid.startswith(prefix):
                    eid = str(row.get("episode_id") or "").strip()
                    if eid:
                        env_ids.add(eid)
    log(
        TAG,
        f"score {job.family}/{job.simulator} seed={job.seed} <- {job.rollout_path}"
        + (f" (env_error_ids={len(env_ids)})" if env_ids else ""),
    )
    run_one(
        join_baseline,
        job.rollout_path,
        metrics=metric_names,
        oss_model=oss_model,
        luar_model=luar_model,
        tau_usi_source=tau_usi_source,
        env_error_episode_ids=env_ids or None,
        no_fit=True,
        merge=merge,
        output_root=runs_dir,
        state_dir=state_dir,
        baseline_paths=base_paths,
        run_dir=run_dir,
        kind=job.kind,
        method=job.method,
        cue_checkpoint=cue_checkpoint,
        cue_session_preprocess=cue_session_preprocess,
    )
    summary_path = run_dir / "summary.json"
    if not summary_path.is_file():
        warn(TAG, f"missing summary for {job.family}/{job.simulator} seed={job.seed}")
        return None
    payload = json.loads(summary_path.read_text(encoding="utf-8"))
    payload["candidate_meta"] = {
        "run_id": job.run_id,
        "family": job.family,
        "method": job.method,
        "kind": job.kind,
        "simulator": job.simulator,
        "seed": job.seed,
        "domain": job.domain,
        "rollout_path": str(job.rollout_path),
        "summary_path": str(summary_path),
    }
    return payload


def _default_score_workers() -> int:
    # Default suite is CPU + small embedding models (shared via lru_cache in-process);
    # thread fan-out reuses one GPU/CPU copy of Style/SimCSE rather than N process copies.
    return max(1, min(32, (os.cpu_count() or 8)))


def _warm_score_encoders(
    metric_names: list[str],
    *,
    luar_model: str = "rrivera1849/LUAR-MUD",
) -> None:
    """Load shared embedding weights once before threaded scoring to avoid init races."""

    names = set(metric_names)
    if "mimicry/wegmann_ava" in names:
        from cue_training.evaluation.metrics.metrics import authenticity as authenticity_metrics

        authenticity_metrics.get_style_encoder()
    if "coverage/styledistance_behavioral" in names:
        from cue_training.evaluation.metrics.metrics.style_encoder import get_styledistance_encoder

        get_styledistance_encoder()


def _score_candidates_parallel(
    *,
    candidates: list[CandidateJob],
    join_baseline: Path,
    base_paths: list[Path],
    runs_dir: Path,
    metric_names: list[str],
    oss_model: str,
    tau_usi_source: str | None,
    state_dir: Path,
    score_workers: int,
    fma_run: Path | None = None,
    luar_model: str = "rrivera1849/LUAR-MUD",
    cue_checkpoint: str | None = None,
    cue_session_preprocess: str = "auto",
    merge: bool = False,
) -> list[dict[str, Any]]:
    workers = max(1, int(score_workers))
    if workers == 1 or len(candidates) <= 1:
        out: list[dict[str, Any]] = []
        for job in candidates:
            payload = _score_job(
                job=job,
                join_baseline=join_baseline,
                base_paths=base_paths,
                runs_dir=runs_dir,
                metric_names=metric_names,
                oss_model=oss_model,
                tau_usi_source=tau_usi_source,
                state_dir=state_dir,
                fma_run=fma_run,
                luar_model=luar_model,
                cue_checkpoint=cue_checkpoint,
                cue_session_preprocess=cue_session_preprocess,
                merge=merge,
            )
            if payload is not None:
                out.append(payload)
        return out

    _warm_score_encoders(metric_names, luar_model=luar_model)
    log(TAG, f"scoring {len(candidates)} candidates with {workers} workers")
    out = []
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futs = {
            pool.submit(
                _score_job,
                job=job,
                join_baseline=join_baseline,
                base_paths=base_paths,
                runs_dir=runs_dir,
                metric_names=metric_names,
                oss_model=oss_model,
                tau_usi_source=tau_usi_source,
                state_dir=state_dir,
                fma_run=fma_run,
                luar_model=luar_model,
                cue_checkpoint=cue_checkpoint,
                cue_session_preprocess=cue_session_preprocess,
                merge=merge,
            ): job
            for job in candidates
        }
        for fut in as_completed(futs):
            job = futs[fut]
            try:
                payload = fut.result()
            except Exception as exc:  # noqa: BLE001
                warn(
                    TAG,
                    f"score failed {job.family}/{job.simulator} seed={job.seed}: {exc}",
                )
                continue
            if payload is not None:
                out.append(payload)
    return out


def run_aggregate(
    *,
    benchmark: str,
    domains: list[str] | str,
    rollouts_root: Path = storage_root() / "outputs" / "rollouts",
    out_dir: Path = storage_root() / "outputs" / "metrics" / "aggregate",
    metrics: list[str] | None = None,
    base_run_id: str = "base",
    simulators: list[str] | None = None,
    methods: set[str] | None = None,
    seeds: list[int] | None = None,
    refit: bool = False,
    oss_model: str = "Qwen/Qwen3-8B",
    luar_model: str = "rrivera1849/LUAR-MUD",
    tau_usi_source: str | None = None,
    domain: str | None = None,  # backward-compat alias
    score_workers: int | None = None,
    fma_run: Path | None = None,
    cue_checkpoint: str | None = None,
    cue_session_preprocess: str = "auto",
    merge: bool = False,
) -> Path:
    """Fit on floor-base sims, score candidates across seeds/domains, emit pooled leaderboard.

    ``merge`` re-runs only ``metrics`` and patches those aggregates into each existing
    ``summary.json`` (other metrics kept), then rebuilds the leaderboard from the union.
    """

    metric_names = list(metrics or DEFAULT_METRICS)
    sims = list(simulators or ["gpt", "llama", "gemini"])
    workers = _default_score_workers() if score_workers is None else max(1, int(score_workers))
    rollouts_root = Path(rollouts_root)
    if isinstance(domains, str):
        domain_list = [domains]
    else:
        domain_list = list(domains)
    if domain and not domain_list:
        domain_list = [domain]
    if not domain_list:
        raise ValueError("at least one domain is required")

    label = domain_list[0] if len(domain_list) == 1 else "all"
    agg_root = Path(out_dir) / f"{benchmark}_{label}"
    runs_dir = agg_root / "_runs"
    agg_root.mkdir(parents=True, exist_ok=True)

    all_summaries: list[dict[str, Any]] = []
    base_paths_by_domain: dict[str, list[Path]] = {}

    for dom in domain_list:
        state_dir = agg_root / "state" / dom
        base_pairs = resolve_base_paths(
            rollouts_root,
            benchmark=benchmark,
            domain=dom,
            base_run_id=base_run_id,
            simulators=sims,
            seeds=seeds,
        )
        if len(base_pairs) < 2:
            raise ValueError(
                f"Need >=2 floor-base simulator rollouts for {benchmark}/{dom} "
                f"(base_run_id={base_run_id}); found {[s for s, _ in base_pairs]}"
            )
        base_paths = [p for _, p in base_pairs]
        sim_labels = [s for s, _ in base_pairs]
        base_paths_by_domain[dom] = base_paths
        join_baseline = base_paths[0]
        log(TAG, f"[{dom}] aggregate fit on {len(base_paths)} floor-base sims: {sim_labels}")
        fit_aggregate_state(
            base_paths,
            state_dir,
            metrics=metric_names,
            refit=refit,
            oss_model=oss_model,
            luar_model=luar_model,
            sim_labels=sim_labels,
            cue_checkpoint=cue_checkpoint,
            cue_session_preprocess=cue_session_preprocess,
        )

        # Human upper bound once per domain (seed-invariant).
        human_path = runs_dir / _HUMAN_LABEL / dom / "rollout.human.jsonl"
        write_human_proxy_rollout(join_baseline, human_path, tau_usi_source=tau_usi_source)
        human_run_dir = runs_dir / _HUMAN_LABEL / dom / "all"
        log(TAG, f"[{dom}] score human upper bound <- {human_path}")
        run_one(
            join_baseline,
            human_path,
            metrics=metric_names,
            oss_model=oss_model,
            luar_model=luar_model,
            tau_usi_source=tau_usi_source,
            no_fit=True,
            merge=merge,
            output_root=runs_dir,
            state_dir=state_dir,
            baseline_paths=base_paths,
            run_dir=human_run_dir,
            kind=_HUMAN_LABEL,
            method=_HUMAN_LABEL,
            cue_checkpoint=cue_checkpoint,
            cue_session_preprocess=cue_session_preprocess,
        )
        human_summary = human_run_dir / "summary.json"
        if human_summary.is_file():
            payload = json.loads(human_summary.read_text(encoding="utf-8"))
            payload["candidate_meta"] = {
                "run_id": _HUMAN_LABEL,
                "family": _HUMAN_LABEL,
                "method": _HUMAN_LABEL,
                "kind": _HUMAN_LABEL,
                "simulator": "—",
                "seed": None,
                "domain": dom,
                "rollout_path": str(human_path),
                "summary_path": str(human_summary),
                "upper_bound": True,
            }
            all_summaries.append(payload)
        else:
            warn(TAG, f"missing human upper-bound summary at {human_summary}")

        candidates = discover_candidates(
            rollouts_root,
            benchmark=benchmark,
            domain=dom,
            methods=methods,
            seeds=seeds,
        )
        if not candidates:
            warn(TAG, f"No complete candidate jobs under {rollouts_root} for {benchmark}/{dom}")
            continue
        log(
            TAG,
            f"[{dom}] scoring {len(candidates)} candidate jobs "
            f"(score_workers={workers})",
        )
        all_summaries.extend(
            _score_candidates_parallel(
                candidates=candidates,
                join_baseline=join_baseline,
                base_paths=base_paths,
                runs_dir=runs_dir / dom,
                metric_names=metric_names,
                oss_model=oss_model,
                tau_usi_source=tau_usi_source,
                state_dir=state_dir,
                score_workers=workers,
                fma_run=fma_run,
                luar_model=luar_model,
                cue_checkpoint=cue_checkpoint,
                cue_session_preprocess=cue_session_preprocess,
                merge=merge,
            )
        )

    if not all_summaries:
        raise ValueError(f"No scored candidates for {benchmark} domains={domain_list}")

    if merge:
        lb_source = metrics_present_in_summaries(all_summaries) or metric_names
    else:
        lb_source = metric_names
    lb_metrics = leaderboard_metric_names(lb_source)
    seed_rows = summaries_to_rows(all_summaries, metric_names=lb_metrics)

    per_domain = pool_across_seeds(seed_rows, metric_names=lb_metrics)
    leaderboard_rows = list(per_domain)
    if benchmark in _MACRO_DOMAIN_BENCHMARKS:
        # For umbrella domains (tau2 customer-service), macro over group domains.
        # For multi-folder domains (simarena writing+math), macro over those folder names.
        macro_rows = macro_across_domains(
            seed_rows, metric_names=lb_metrics, domains=domain_list
        )
        leaderboard_rows.extend(macro_rows)

    rows, markdown = build_leaderboard(leaderboard_rows, metric_names=lb_metrics)
    # Also keep raw per-seed rows for debugging / re-pooling.
    leaderboard_json = {
        "benchmark": benchmark,
        "domains": domain_list,
        "seeds": seeds,
        "base_run_id": base_run_id,
        "base_paths_by_domain": {
            d: [str(p) for p in ps] for d, ps in base_paths_by_domain.items()
        },
        "metrics": lb_metrics,
        "n_summaries": len(all_summaries),
        "includes_human_upper_bound": any(
            (s.get("candidate_meta") or {}).get("upper_bound") for s in all_summaries
        ),
        "pooling": {
            "across": "seeds",
            "error_bars": "stdev(seed_means)",
            "domain_macro": (
                "equal-weight mean of domains per seed, then mean ± stdev over those macros"
                if benchmark in _MACRO_DOMAIN_BENCHMARKS
                else None
            ),
        },
        "rows": rows,
        "per_seed_rows": seed_rows,
    }
    json_path = agg_root / "leaderboard.json"
    md_path = agg_root / "leaderboard.md"
    json_path.write_text(json.dumps(leaderboard_json, indent=2), encoding="utf-8")
    md_path.write_text(markdown, encoding="utf-8")
    log(TAG, f"leaderboard -> {md_path} ({len(rows)} pooled rows from {len(all_summaries)} summaries)")
    return md_path


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Fit metrics on floor-base simulator rollouts, score every discovered job across "
            "seeds, and emit a leaderboard with seed-mean ± stdev(seed means). For tau2/simarena "
            "also emits equal-weight domain macros (macro per seed first) with stdev over those."
        )
    )
    parser.add_argument("--benchmark", required=True, help="e.g. simulatorarena | tau2 | prism")
    parser.add_argument(
        "--domain",
        default="",
        help="Single domain (backward compatible). Prefer --domains for several.",
    )
    parser.add_argument(
        "--domains",
        default="",
        help="Comma-separated domains (e.g. writing,math). Overrides --domain.",
    )
    parser.add_argument(
        "--rollouts-root", type=Path, default=storage_root() / "outputs" / "rollouts"
    )
    parser.add_argument(
        "--out-dir", type=Path, default=storage_root() / "outputs" / "metrics" / "aggregate"
    )
    parser.add_argument("--metrics", default="default", help="'default' or comma-separated paper metric names.")
    parser.add_argument(
        "--base-run-id",
        default="base",
        help=(
            "Run-id folder (or prefix) for floor-base jobs. Also resolves "
            "{id}-seed-N and multiseed baseline runs that contain baseline/base/{sim}."
        ),
    )
    parser.add_argument(
        "--simulators",
        default="gpt,llama,gemini",
        help="Comma-separated floor-base simulator names to pool for the fit.",
    )
    parser.add_argument(
        "--methods",
        default="",
        help="Optional comma-separated run_id / method / family filter (default: all).",
    )
    parser.add_argument(
        "--seeds",
        default="0,1,2",
        help=(
            "Comma-separated seed indices under rollouts_root/seed-N/ (default 0,1,2). "
            "Empty string = auto-detect all seed-* dirs, or flat layout if none."
        ),
    )
    parser.add_argument("--refit", action="store_true")
    parser.add_argument(
        "--merge",
        action="store_true",
        help="Re-run only --metrics and patch those aggregates into each existing summary.json "
             "(keep other metrics), then rebuild the leaderboard from the union. "
             "Example: --metrics turing,paired_audit,mrr3 --merge",
    )
    parser.add_argument(
        "--score-workers",
        type=int,
        default=None,
        help=(
            "Thread workers for scoring candidate jobs after the shared fit "
            f"(default: min(32, cpu_count)={_default_score_workers()}). Set 1 for serial."
        ),
    )
    parser.add_argument("--oss-model", default="Qwen/Qwen3-8B")
    parser.add_argument(
        "--luar-model",
        default="rrivera1849/LUAR-MUD",
        help="LUAR checkpoint for luar alignment / mimicry / coverage metrics.",
    )
    parser.add_argument("--tau-usi-source", default=None)
    parser.add_argument(
        "--fma-run",
        type=Path,
        default=None,
        help="FMA run dir (or id under $CUE_STORAGE_ROOT/fma). Environment Error tags "
             "are treated as successes for pairwise Success F1 vs human.",
    )
    args = parser.parse_args(argv)

    metric_names = _parse_metrics(args.metrics)
    sims = [s.strip() for s in str(args.simulators).split(",") if s.strip()]
    method_filter = {m.strip() for m in str(args.methods).split(",") if m.strip()} or None
    domains_arg = [d.strip() for d in str(args.domains).split(",") if d.strip()]
    if not domains_arg and args.domain:
        domains_arg = [str(args.domain).strip()]
    if not domains_arg:
        raise SystemExit("provide --domain or --domains")

    seeds_raw = str(args.seeds).strip()
    if seeds_raw == "":
        seed_list: list[int] | None = None
    else:
        seed_list = [int(x.strip()) for x in seeds_raw.split(",") if x.strip()]

    try:
        run_aggregate(
            benchmark=args.benchmark,
            domains=domains_arg,
            rollouts_root=args.rollouts_root,
            out_dir=args.out_dir,
            metrics=metric_names,
            base_run_id=args.base_run_id,
            simulators=sims,
            methods=method_filter,
            seeds=seed_list,
            refit=args.refit,
            oss_model=args.oss_model,
            luar_model=args.luar_model,
            tau_usi_source=args.tau_usi_source,
            score_workers=args.score_workers,
            fma_run=args.fma_run,
            merge=args.merge,
        )
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc


if __name__ == "__main__":
    main()
