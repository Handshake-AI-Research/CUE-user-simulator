from __future__ import annotations

import argparse
import json
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Any

from cue_training.utils.config import storage_root
from cue_training.evaluation.metrics import cache
from cue_training.evaluation.metrics.data import load_baseline, load_baseline_paths, load_episodes
from cue_training.evaluation.metrics.display import render_markdown
from cue_training.evaluation.metrics.metrics import api as api_metrics
from cue_training.evaluation.metrics.metrics import authenticity as authenticity_metrics
from cue_training.evaluation.metrics.metrics import coverage as coverage_metrics
from cue_training.evaluation.metrics.metrics import local as local_metrics
from cue_training.evaluation.metrics.policy import skip_ava_mrr, skip_paired_audit
from cue_training.evaluation.metrics.stats import Aggregate, aggregate, aggregate_from_json
from cue_training.runlog.log import log, warn


@contextmanager
def _timed(label: str):
    """Log ``label`` start + elapsed seconds, so a slow/blocked step is visible in the log."""

    log("metrics", f"{label} ...")
    start = time.monotonic()
    try:
        yield
    finally:
        log("metrics", f"{label} done in {time.monotonic() - start:.1f}s")


# Default suite (shared by ``metrics`` and ``metrics auto``).
DEFAULT_METRICS = [
    "env/tau2_success_rate",
    "env/tau2_task_success",
    "classifier/sim2real",
    "judge/turing_sonnet_qwen",
    "mimicry/wegmann_ava",
    "mimicry/paired_audit",
    "coverage/sim2real_behavioral",
    "coverage/styledistance_behavioral",
]

ALL_METRICS = list(DEFAULT_METRICS)

# Metrics that need the full multi-arm episode list (not a single arm bucket).
_GLOBAL_METRICS = {
    "coverage/styledistance_behavioral",
    "coverage/sim2real_behavioral",
}

_TURING_METRICS = {"judge/turing_sonnet_qwen", "mimicry/paired_audit"}


def _slug(path: Path) -> str:
    stem = path.stem.replace(".", "_")
    parent = path.parent.name.replace(".", "_")
    return f"{parent}_{stem}" if parent else stem


def _model_slug(path: Path) -> str:
    """Just the model directory name, e.g. ``.../gpt/rollout.tau2.jsonl`` -> ``gpt``."""

    parent = path.parent.name.replace(".", "_")
    return parent or path.stem.replace(".", "_")


def _run_id(baseline: Path, candidate: Path | None) -> str:
    base = _model_slug(baseline)
    if candidate is None:
        return base
    other = _model_slug(candidate)
    # Self-comparisons are the common case; do not repeat the model name for them.
    return base if other == base else f"{base}__{other}"


def _default_run_dir(
    baseline: Path,
    candidate: Path | None,
    root: Path = storage_root() / "outputs" / "metrics",
) -> Path:
    return root / _run_id(baseline, candidate)


def _parse_metrics(raw: str) -> list[str]:
    if raw in ("default", "defaults", "all", ""):
        return list(DEFAULT_METRICS)
    aliases = {
        "ava": "mimicry/wegmann_ava",
        "wegmann_ava": "mimicry/wegmann_ava",
        "sim2real_classifier": "classifier/sim2real",
        "task_success": "env/tau2_task_success",
        "env_success": "env/tau2_success_rate",
        "success_rate": "env/tau2_success_rate",
        "styledistance_behavioral": "coverage/styledistance_behavioral",
        "sim2real_behavioral": "coverage/sim2real_behavioral",
        "sim2real_coverage": "coverage/sim2real_behavioral",
        "coverage": "coverage/styledistance_behavioral,coverage/sim2real_behavioral",
        "turing": "judge/turing_sonnet_qwen",
        "paired_audit": "mimicry/paired_audit",
    }
    allowed = set(ALL_METRICS)
    out: list[str] = []
    for part in raw.split(","):
        key = part.strip()
        if not key:
            continue
        if key in ("default", "defaults", "all"):
            for name in DEFAULT_METRICS:
                if name not in out:
                    out.append(name)
            continue
        expanded = aliases.get(key, key)
        for item in expanded.split(","):
            name = item.strip()
            if not name:
                continue
            if name not in allowed:
                raise ValueError(
                    f"unsupported metric {name!r}; available: {sorted(allowed)}"
                )
            if name not in out:
                out.append(name)
    return out


def _state_config(
    metrics: list[str],
    oss_model: str,
    *,
    luar_model: str = "rrivera1849/LUAR-MUD",
    cue_checkpoint: str | None = None,
) -> dict:
    # Bump version when cached baseline-state schema changes (not when only a scorer formula changes).
    # v9: sim2real probe is a RandomForest over 19-D PPol fingerprints, not an 11-D lexical LR,
    # so a stored v8 probe has both the wrong model and the wrong input width -- must refit.
    # v10: also fit a StyleDistance RandomForest discriminator (styledistance_probe_state.pkl).
    # v11: optional SimCSE / Wegmann / LUAR RF discriminators for extended naturalness.
    # v12: Turing few-shot/eval ids + optional CUE Stage-1 probe for --cue.
    # v13: Qwen-0.6B LoRA + logistic regression naturalness probe.
    cfg: dict = {"metrics": sorted(metrics), "version": 13}
    if "mimicry/wegmann_ava" in metrics:
        cfg["authenticity"] = {
            "simcse": "princeton-nlp/sup-simcse-roberta-large",
            "style": "AnnaWegmann/Style-Embedding",
            "paper": "wang2025usp",
        }
    if set(metrics) & _TURING_METRICS:
        cfg["turing"] = {"eval_n": 60, "fewshot_per_class": 4}
    return cfg


def _fit_state(
    *,
    baseline_path: Path,
    baseline,
    cache_dir: Path,
    metrics: list[str],
    refit: bool,
    oss_model: str,
    baseline_paths: list[Path] | None = None,
    config_extra: dict | None = None,
    luar_model: str = "rrivera1849/LUAR-MUD",
    cue_checkpoint: str | None = None,
    cue_session_preprocess: str = "auto",
) -> None:
    config = _state_config(
        metrics, oss_model, luar_model=luar_model, cue_checkpoint=cue_checkpoint
    )
    if config_extra:
        config.update(config_extra)
    paths = list(baseline_paths) if baseline_paths else [baseline_path]
    if not refit and cache.manifest_matches(cache_dir, baselines=paths, config=config):
        log("metrics", f"reusing baseline state -> {cache_dir}")
        return
    log("metrics", f"fitting baseline state -> {cache_dir} ({len(baseline)} baseline episodes)")
    with _timed("fit local state"):
        local_metrics.fit_baseline_state(baseline, cache_dir)
    if set(metrics) & _TURING_METRICS:
        with _timed("fit turing few-shot / eval ids"):
            api_metrics.fit_turing_state(baseline, cache_dir)
    cache.write_manifest(cache_dir, baselines=paths, config=config)


def fit_aggregate_state(
    baseline_paths: list[Path],
    cache_dir: Path,
    *,
    metrics: list[str],
    refit: bool = False,
    oss_model: str = "Qwen/Qwen3-8B",
    luar_model: str = "rrivera1849/LUAR-MUD",
    sim_labels: list[str] | None = None,
    seed: int = 0,
    cue_checkpoint: str | None = None,
    cue_session_preprocess: str = "auto",
) -> dict:
    """Fit discriminators on floor-sim JSONLs with one seeded sim negative per human episode."""

    paths = [Path(p) for p in baseline_paths]
    if len(paths) < 2:
        raise ValueError(f"aggregate fit needs >=2 baseline paths, got {len(paths)}")
    for path in paths:
        if not path.is_file():
            raise ValueError(f"aggregate baseline missing: {path}")
    labels = list(sim_labels) if sim_labels is not None else [p.parent.name for p in paths]
    baseline = load_baseline_paths(paths, sim_labels=labels, seed=seed)
    if not baseline:
        raise ValueError(f"No usable episodes in aggregate baselines: {paths}")
    _fit_state(
        baseline_path=paths[0],
        baseline=baseline,
        cache_dir=cache_dir,
        metrics=metrics,
        refit=refit,
        oss_model=oss_model,
        luar_model=luar_model,
        baseline_paths=paths,
        cue_checkpoint=cue_checkpoint,
        cue_session_preprocess=cue_session_preprocess,
        config_extra={
            "aggregate_negatives": "one_sim_per_episode",
            "aggregate_seed": int(seed),
        },
    )
    return baseline


def _run_metrics(
    metrics: list[str],
    episodes,
    cache_dir: Path,
    *,
    tau_usi_source: str | None = None,
    env_error_episode_ids: set[str] | None = None,
    luar_model: str = "rrivera1849/LUAR-MUD",
    kind: str | None = None,
    method: str | None = None,
    arm: str | None = None,
    baseline=None,
    cue_checkpoint: str | None = None,
    cue_session_preprocess: str = "auto",
) -> list[Aggregate]:
    per_group = [m for m in metrics if m not in _GLOBAL_METRICS]
    out: list[Aggregate] = []
    for i, metric in enumerate(per_group, 1):
        start = time.monotonic()
        log("metrics", f"[{i}/{len(per_group)}] running {metric} ({len(episodes)} episodes) ...")
        group_arm = arm or (episodes[0].arm if episodes else None)
        if metric == "env/tau2_task_success":
            out.append(
                local_metrics.tau2_task_success(
                    episodes,
                    tau_usi_source=tau_usi_source,
                    env_error_episode_ids=env_error_episode_ids,
                )
            )
        elif metric == "env/tau2_success_rate":
            out.append(
                local_metrics.tau2_success_rate(
                    episodes,
                    tau_usi_source=tau_usi_source,
                    env_error_episode_ids=env_error_episode_ids,
                )
            )
        elif metric == "classifier/sim2real":
            out.append(local_metrics.sim2real_classifier(episodes, cache_dir))
        elif metric == "mimicry/wegmann_ava":
            reason = skip_ava_mrr(kind=kind, method=method, arm=group_arm)
            if reason:
                out.append(aggregate(metric, [], {"skipped": reason}))
            else:
                agg = authenticity_metrics.run_authenticity_metric(
                    "mimicry/wegmann_ava", episodes, luar_model=luar_model
                )
                if agg.metric_name != metric:
                    agg = Aggregate(
                        metric_name=metric,
                        mean=agg.mean,
                        standard_deviation=agg.standard_deviation,
                        confidence_interval=agg.confidence_interval,
                        sample_size=agg.sample_size,
                        extras=agg.extras,
                    )
                out.append(agg)
        elif metric == "judge/turing_sonnet_qwen":
            out.append(
                api_metrics.turing_sonnet_qwen(
                    episodes, cache_dir, baseline=baseline, human_control=(method == "human")
                )
            )
        elif metric == "mimicry/paired_audit":
            reason = skip_paired_audit(kind=kind, method=method, arm=group_arm)
            if reason:
                out.append(aggregate(metric, [], {"skipped": reason}))
            else:
                out.append(
                    api_metrics.paired_audit_fidelity(
                        episodes, cache_dir, human_control=(method == "human")
                    )
                )
        else:
            out.append(Aggregate(metric_name=metric, mean=float("nan"), sample_size=0, extras={"error": "unknown metric"}))
        log("metrics", f"[{i}/{len(per_group)}] {metric} done in {time.monotonic() - start:.1f}s")
    return out


def _run_global_metrics(
    metrics: list[str],
    episodes,
    baseline,
    *,
    luar_model: str = "rrivera1849/LUAR-MUD",
    kind: str | None = None,
    method: str | None = None,
    cue_checkpoint: str | None = None,
    cue_session_preprocess: str = "auto",
) -> dict[str, list[Aggregate]]:
    """Run multi-arm metrics; return aggregates keyed by group (arm or arm/domain)."""

    by_group: dict[str, list[Aggregate]] = {}
    for metric in metrics:
        if metric not in _GLOBAL_METRICS:
            continue
        start = time.monotonic()
        log("metrics", f"running global {metric} ({len(episodes)} episodes) ...")
        if metric == "coverage/styledistance_behavioral":
            results = coverage_metrics.coverage_styledistance_behavioral(episodes)
        elif metric == "coverage/sim2real_behavioral":
            results = coverage_metrics.coverage_sim2real_behavioral(episodes)
        else:
            results = {}
        for group, agg in results.items():
            by_group.setdefault(group, []).append(agg)
        log("metrics", f"global {metric} done in {time.monotonic() - start:.1f}s ({len(results)} groups)")
    return by_group


def _write_summary_json(
    path: Path,
    *,
    run_id: str,
    baseline: Path,
    candidate: Path | None,
    aggregates_by_group: dict[str, list[Aggregate]],
    n_by_group: dict[str, int],
    group_meta: dict[str, dict[str, str]],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload: dict[str, Any] = {
        "run": {
            "run_id": run_id,
            "engine": "native-cue",
            "baseline": str(baseline),
            "candidate": str(candidate or baseline),
            "groups": list(aggregates_by_group),
            "arms": sorted({m["arm"] for m in group_meta.values()}),
            "domains": sorted({m["domain"] for m in group_meta.values() if m["domain"] != "all"}),
        },
        # aggregates_by_arm kept as alias for existing consumers (keys may be arm or arm/domain).
        "aggregates_by_arm": {g: [agg.to_json() for agg in aggs] for g, aggs in aggregates_by_group.items()},
        "aggregates_by_group": {g: [agg.to_json() for agg in aggs] for g, aggs in aggregates_by_group.items()},
        "group_meta": group_meta,
        "telemetry_stats": {
            "episodes_total": sum(n_by_group.values()),
            "episodes_by_arm": n_by_group,
            "episodes_by_group": n_by_group,
        },
    }
    if len(aggregates_by_group) == 1:  # keep the single-group schema flat for existing consumers
        payload["aggregates"] = next(iter(payload["aggregates_by_group"].values()))
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _load_summary_aggregates(path: Path) -> tuple[dict[str, list[Aggregate]], dict[str, int], dict[str, dict[str, str]]] | None:
    """Load aggregates_by_group (+ n/meta) from an existing summary.json, or None if absent/unusable."""

    if not path.is_file():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    raw = payload.get("aggregates_by_group") or payload.get("aggregates_by_arm")
    if not isinstance(raw, dict) or not raw:
        return None
    by_group = {
        str(g): [aggregate_from_json(a) for a in aggs if isinstance(a, dict)]
        for g, aggs in raw.items()
        if isinstance(aggs, list)
    }
    group_meta = payload.get("group_meta") if isinstance(payload.get("group_meta"), dict) else {}
    group_meta = {str(k): dict(v) for k, v in group_meta.items() if isinstance(v, dict)}
    tel = payload.get("telemetry_stats") if isinstance(payload.get("telemetry_stats"), dict) else {}
    n_by_group = tel.get("episodes_by_group") or tel.get("episodes_by_arm") or {}
    n_by_group = {str(k): int(v) for k, v in n_by_group.items()} if isinstance(n_by_group, dict) else {}
    for g, aggs in by_group.items():
        n_by_group.setdefault(g, next((a.sample_size for a in aggs if a.sample_size), 0))
        group_meta.setdefault(g, {"arm": g.split("/")[0], "domain": g.split("/")[1] if "/" in g else "all"})
    return by_group, n_by_group, group_meta


def _merge_aggregates(
    existing: dict[str, list[Aggregate]],
    updated: dict[str, list[Aggregate]],
) -> dict[str, list[Aggregate]]:
    """Replace metrics present in ``updated``; keep all other aggregates from ``existing``."""

    out: dict[str, list[Aggregate]] = {g: list(aggs) for g, aggs in existing.items()}
    for group, new_aggs in updated.items():
        by_name = {a.metric_name: a for a in out.get(group, [])}
        order = [a.metric_name for a in out.get(group, [])]
        for agg in new_aggs:
            if agg.metric_name not in by_name:
                order.append(agg.metric_name)
            by_name[agg.metric_name] = agg
        out[group] = [by_name[name] for name in order if name in by_name]
    return out


def _group_episodes(episodes) -> tuple[dict[str, list], dict[str, dict[str, str]]]:
    """Bucket episodes by arm, and by arm/domain when multiple domains are present.

    Tau2 mixes airline+retail; SimulatorArena may mix writing+math. Single-domain runs
    keep the previous arm-only keys so reports stay compact.
    """

    domains = {ep.domain for ep in episodes if ep.domain and ep.domain != "all"}
    split_domain = len(domains) > 1
    by_group: dict[str, list] = {}
    group_meta: dict[str, dict[str, str]] = {}
    for ep in episodes:
        if split_domain:
            key = f"{ep.arm}/{ep.domain}"
            group_meta[key] = {"arm": ep.arm, "domain": ep.domain}
        else:
            key = ep.arm
            domain = next(iter(domains), "all")
            group_meta[key] = {"arm": ep.arm, "domain": domain}
        by_group.setdefault(key, []).append(ep)
    return by_group, group_meta


def run_one(
    baseline_path: Path,
    candidate_path: Path | None = None,
    *,
    metrics: list[str],
    oss_model: str = "Qwen/Qwen3-8B",
    luar_model: str = "rrivera1849/LUAR-MUD",
    tau_usi_source: str | None = None,
    env_error_episode_ids: set[str] | None = None,
    refit: bool = False,
    no_fit: bool = False,
    merge: bool = False,
    output_root: Path = storage_root() / "outputs" / "metrics",
    state_dir: Path | None = None,
    baseline_paths: list[Path] | None = None,
    run_dir: Path | None = None,
    kind: str | None = None,
    method: str | None = None,
    cue_checkpoint: str | None = None,
    cue_session_preprocess: str = "auto",
) -> None:
    """Fit (or reuse) the baseline state and score one (baseline, candidate) pair, writing the
    summary.md/json under ``output_root`` (default ``$CUE_STORAGE_ROOT/outputs/metrics``).

    ``state_dir`` reuses a pre-fitted baseline state from an explicit directory (decoupled from
    ``output_root``/slug), so repeated runs don't refit; reuse is gated on the baseline content
    hash + metric config matching (pair with ``no_fit`` to require it).

    ``baseline_paths`` (optional) is the set of floor-sim JSONLs the fitted state was built
    from -- used by the aggregate eval so ``no_fit`` checks the combined multi-sim hash rather
    than the single human-join ``baseline_path``.

    ``run_dir`` (optional) writes summaries to an explicit directory instead of the default
    ``output_root / <model_slug>`` layout (used by the aggregate leaderboard scorer).

    ``merge`` re-runs only ``metrics`` and patches those aggregates into an existing
    ``summary.json`` (other metrics kept). Use with ``--metrics <one>`` to refresh a single
    score without wiping the rest of the report.

    ``kind`` / ``method`` drive skip rules (AVA/MRR, paired audit, MAUVE). ``cue_checkpoint``
    enables Stage-1 CUE re-encode for cue-space embedding metrics.
    """

    output_root = Path(output_root)
    run_dir = Path(run_dir) if run_dir is not None else _default_run_dir(baseline_path, candidate_path, output_root)
    # Fitted state stays keyed by the full rollout slug, not the model name: two rollouts
    # for one model must not share a cache dir, or they refit over each other.
    cache_dir = Path(state_dir) if state_dir else output_root / _slug(baseline_path) / "state"
    out_md = run_dir / "summary.md"
    out_json = run_dir / "summary.json"
    fit_paths = list(baseline_paths) if baseline_paths else [baseline_path]

    if method is None and candidate_path is not None:
        method = candidate_path.parent.name
    if kind is None and candidate_path is not None:
        kind = candidate_path.parent.parent.name

    baseline = load_baseline(baseline_path)
    if not baseline:
        raise ValueError(f"No usable baseline episodes found in {baseline_path}")
    config = _state_config(
        metrics, oss_model, luar_model=luar_model, cue_checkpoint=cue_checkpoint
    )
    if no_fit and not cache.manifest_matches(cache_dir, baselines=fit_paths, config=config):
        raise ValueError(f"Missing/incompatible baseline state at {cache_dir}; run without no_fit.")
    if not no_fit:
        _fit_state(
            baseline_path=baseline_path,
            baseline=baseline,
            cache_dir=cache_dir,
            metrics=metrics,
            refit=refit,
            oss_model=oss_model,
            luar_model=luar_model,
            baseline_paths=fit_paths,
            cue_checkpoint=cue_checkpoint,
            cue_session_preprocess=cue_session_preprocess,
        )

    episodes, _baseline = load_episodes(baseline_path, candidate_path)
    if not episodes:
        raise ValueError(f"No joined episodes to score for {candidate_path or baseline_path}.")
    # Score each rollout arm (and domain, when mixed) separately.
    by_group, group_meta = _group_episodes(episodes)
    n_by_group = {g: len(eps) for g, eps in by_group.items()}
    aggregates_by_group: dict[str, list[Aggregate]] = {}
    for group, eps in by_group.items():
        meta = group_meta[group]
        log("metrics", f"scoring arm '{meta['arm']}' domain '{meta['domain']}' ({len(eps)} episodes)")
        aggregates_by_group[group] = _run_metrics(
            metrics,
            eps,
            cache_dir,
            tau_usi_source=tau_usi_source,
            env_error_episode_ids=env_error_episode_ids,
            luar_model=luar_model,
            kind=kind,
            method=method,
            arm=meta["arm"],
            baseline=baseline,
            cue_checkpoint=cue_checkpoint,
            cue_session_preprocess=cue_session_preprocess,
        )

    # Mimicry / Coverage need the full multi-arm list.
    if set(metrics) & _GLOBAL_METRICS:
        global_by_group = _run_global_metrics(
            metrics,
            episodes,
            baseline,
            luar_model=luar_model,
            kind=kind,
            method=method,
            cue_checkpoint=cue_checkpoint,
            cue_session_preprocess=cue_session_preprocess,
        )
        for group, aggs in global_by_group.items():
            aggregates_by_group.setdefault(group, []).extend(aggs)
            n_by_group.setdefault(group, n_by_group.get(group, 0) or len(by_group.get(group, [])))
            if group not in group_meta:
                arm = group.split("/")[0]
                # Arm-only keys (e.g. MAUVE pooled across airline+retail) must stay
                # domain="all". Inferring the first subdomain stamped arm-pooled rows as
                # "airline", which then overwrote airline in the domain macro.
                domain = group.split("/")[1] if "/" in group else "all"
                group_meta[group] = {"arm": arm, "domain": domain}

    if merge:
        loaded = _load_summary_aggregates(out_json)
        if loaded is None:
            warn("metrics", f"--merge requested but no usable summary at {out_json}; writing fresh results only")
        else:
            existing, existing_n, existing_meta = loaded
            aggregates_by_group = _merge_aggregates(existing, aggregates_by_group)
            for g, n in existing_n.items():
                n_by_group.setdefault(g, n)
            for g, meta in existing_meta.items():
                group_meta.setdefault(g, meta)
            log("metrics", f"merged {metrics} into existing {out_json}")

    run_id = _run_id(baseline_path, candidate_path)
    markdown = render_markdown(
        run_id=run_id,
        baseline_path=baseline_path,
        candidate_path=candidate_path,
        aggregates_by_group=aggregates_by_group,
        n_by_group=n_by_group,
        group_meta=group_meta,
    )
    out_md.parent.mkdir(parents=True, exist_ok=True)
    out_md.write_text(markdown, encoding="utf-8")
    _write_summary_json(
        out_json,
        run_id=run_id,
        baseline=baseline_path,
        candidate=candidate_path,
        aggregates_by_group=aggregates_by_group,
        n_by_group=n_by_group,
        group_meta=group_meta,
    )
    log("metrics", f"wrote {out_md} and {out_json}")


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Run native CUE metrics over env rollout JSONL files.")
    parser.add_argument("--baseline", required=True, type=Path, help="Baseline rollout.tau2.jsonl (also provides human references).")
    parser.add_argument("--candidate", type=Path, default=None, help="Candidate rollout.tau2.jsonl; omitted = score baseline against itself.")
    parser.add_argument(
        "--metrics",
        default="default",
        help="Comma-separated metric names, or 'default' for the eight paper metrics.",
    )
    parser.add_argument("--method", default=None, help="Method name for skip rules (else inferred from candidate path).")
    parser.add_argument("--kind", default=None, help="Kind (baseline|cue|human) for skip rules.")
    parser.add_argument("--refit", action="store_true", help="Rebuild baseline state even if cache matches.")
    parser.add_argument("--no-fit", action="store_true", help="Do not fit missing baseline state; fail if cache is absent/incompatible.")
    parser.add_argument(
        "--merge",
        action="store_true",
        help="Re-run only --metrics and patch those aggregates into the existing summary.json/md "
             "(keep other metrics). Example: --metrics ava --merge --no-fit.",
    )
    parser.add_argument("--tau-usi-source", default=None, help="Local tau_bench_tasks_unified.json for tau2 human rewards (default: HF cmu-lti/tau-usi).")
    parser.add_argument("--oss-model", default="Qwen/Qwen3-8B", help="Unused for the default metric set; retained for fit API compatibility.")
    parser.add_argument("--out-dir", type=Path, default=storage_root() / "outputs" / "metrics",
                        help="Base output dir for summaries + fitted state (default $CUE_STORAGE_ROOT/outputs/metrics). "
                             "A per-run subfolder is created under it.")
    parser.add_argument("--state-dir", type=Path, default=None,
                        help="Reuse a pre-fitted baseline-state directory (skips refitting when the "
                             "baseline content + metrics match). Combine with --no-fit to require it.")
    args = parser.parse_args(argv)

    metrics = _parse_metrics(args.metrics)
    try:
        run_one(
            args.baseline, args.candidate, metrics=metrics,
            oss_model=args.oss_model,
            tau_usi_source=args.tau_usi_source, refit=args.refit,
            no_fit=args.no_fit, merge=args.merge, output_root=args.out_dir, state_dir=args.state_dir,
            kind=args.kind, method=args.method,
        )
    except ValueError as exc:
        raise SystemExit(str(exc))


if __name__ == "__main__":
    main()
