"""CLI: ``cue-fma ingest|batch|review|tag|tvd|plot|discover``."""

from __future__ import annotations

import argparse
import json
from glob import glob
from pathlib import Path

from cue_training.evaluation.fma.paths import (
    default_fma_root,
    default_model,
    ensure_run,
    next_batch_index,
    resolve_run_dir,
)
from cue_training.evaluation.fma.propose import DEFAULT_WORKERS


def _cmd_ingest(args: argparse.Namespace) -> None:
    from cue_training.evaluation.fma.ingest import ingest

    sim_paths: list[Path] = []
    missing: list[str] = []
    for pattern in args.sim or []:
        matches = sorted(glob(pattern, recursive=True))
        if matches:
            sim_paths.extend(Path(m) for m in matches)
        elif Path(pattern).is_file():
            sim_paths.append(Path(pattern))
        else:
            missing.append(pattern)
    if missing:
        raise SystemExit(
            "no rollout files matched --sim pattern(s):\n  "
            + "\n  ".join(missing)
            + "\n(use a path that exists, or a glob with ** and recursive expansion)"
        )
    if args.sim and not sim_paths:
        raise SystemExit("no --sim rollout files to ingest")
    out = resolve_run_dir(args.out)
    result = ingest(
        out_dir=out,
        human_path=Path(args.human) if args.human else None,
        sim_paths=sim_paths,
        rewards_source=args.rewards,
        tasks_dir=args.tasks_dir,
        arms=args.arms,
    )
    print(json.dumps(result, indent=2))


def _cmd_batch(args: argparse.Namespace) -> None:
    from cue_training.evaluation.fma.propose import propose_batch
    from cue_training.evaluation.fma.sample import sample_batch

    run_dir = ensure_run(resolve_run_dir(args.run))
    batch = int(args.batch) if args.batch is not None else next_batch_index(run_dir)
    sampled = sample_batch(run_dir, batch=batch, n=int(args.n), seed=int(args.seed))
    print(json.dumps({"sample": sampled}, indent=2))
    if args.sample_only:
        return
    proposed = propose_batch(
        run_dir, batch=batch, model=args.model, workers=int(args.workers)
    )
    print(json.dumps({"propose": proposed}, indent=2))


def _cmd_discover(args: argparse.Namespace) -> None:
    from cue_training.evaluation.fma.propose import discover_classes

    run_dir = resolve_run_dir(args.run)
    result = discover_classes(run_dir, batch=args.batch, model=args.model)
    print(json.dumps({k: v for k, v in result.items() if k != "raw"}, indent=2))


def _cmd_review(args: argparse.Namespace) -> None:
    from cue_training.evaluation.fma.review_server import serve
    from cue_training.evaluation.fma.sample import sample_uncategorized_batch

    run_dir = ensure_run(resolve_run_dir(args.run))
    batch = int(args.batch) if args.batch is not None else None
    if args.uncategorized:
        if batch is None:
            batch = next_batch_index(run_dir)
        prepared = sample_uncategorized_batch(
            run_dir,
            batch=batch,
            n=int(args.n) if args.n is not None else None,
            seed=int(args.seed),
            label=str(args.label),
        )
        print(json.dumps({"uncategorized_batch": prepared}, indent=2))
        if prepared["n_sampled"] == 0:
            raise SystemExit(
                f"no remaining {args.label!r} tags to review (all already decided, or tagged.jsonl has none)"
            )
    elif batch is None:
        raise SystemExit("cue-fma review requires --batch, or pass --uncategorized")
    serve(str(run_dir), int(batch), host=args.host, port=int(args.port))


def _cmd_tag(args: argparse.Namespace) -> None:
    from cue_training.evaluation.fma.tag import tag_remainder

    result = tag_remainder(
        resolve_run_dir(args.run),
        model=args.model,
        workers=int(args.workers),
        force=bool(args.force),
        relabel=list(args.relabel or []),
        reasoning_effort=(args.reasoning_effort or None),
    )
    print(json.dumps(result, indent=2))


def _exclude_labels_from_args(args: argparse.Namespace) -> list[str]:
    from cue_training.evaluation.fma.sim_errors import ALL_SIMULATOR_ERROR_LABELS

    labels = list(args.exclude_label or [])
    if getattr(args, "no_sim_error", False):
        labels.extend(ALL_SIMULATOR_ERROR_LABELS)
    # Preserve order, drop empties/dupes.
    seen: set[str] = set()
    out: list[str] = []
    for lab in labels:
        name = str(lab).strip()
        if name and name not in seen:
            seen.add(name)
            out.append(name)
    return out


def _merge_labels_from_args(args: argparse.Namespace) -> dict[str, str]:
    from cue_training.evaluation.fma.sim_errors import (
        USER_ERROR_BUCKET_SOURCES,
        USER_ERROR_BUCKET_TARGET,
    )

    merge: dict[str, str] = {}
    if getattr(args, "leakage_as_sim_error", False) and not getattr(
        args, "user_error", False
    ):
        merge["User Data Leakage"] = "User Identity / Task Derailment"
    if getattr(args, "user_error", False):
        for src in USER_ERROR_BUCKET_SOURCES:
            merge[src] = USER_ERROR_BUCKET_TARGET
    for item in getattr(args, "merge_label", None) or []:
        raw = str(item)
        if "=" not in raw:
            raise SystemExit(f"--merge-label expects OLD=NEW, got {raw!r}")
        old, new = raw.split("=", 1)
        old, new = old.strip(), new.strip()
        if not old or not new:
            raise SystemExit(f"--merge-label expects OLD=NEW, got {raw!r}")
        merge[old] = new
    return merge


def _cmd_plot(args: argparse.Namespace) -> None:
    from cue_training.evaluation.fma.plots.plot_failure_types import run_plot

    run_dir = resolve_run_dir(args.run)
    out = Path(args.out_dir).expanduser() if args.out_dir else None
    result = run_plot(
        run_dir,
        out_dir=out,
        exclude_labels=_exclude_labels_from_args(args),
        merge_labels=_merge_labels_from_args(args),
    )
    print(json.dumps(result, indent=2))


def _cmd_scatter(args: argparse.Namespace) -> None:
    from cue_training.evaluation.fma.plots.plot_metrics_scatter import run_scatter

    result = run_scatter(
        leaderboard=Path(args.leaderboard),
        distributions=Path(args.distributions),
        out_dir=Path(args.out_dir),
        stem=args.stem,
        no_human=bool(args.no_human),
        no_plot=bool(args.no_plot),
    )
    if result.get("missing"):
        print(f"Skipped {len(result['missing'])} source(s):")
        for line in result["missing"]:
            print(f"  - {line}")
    print(json.dumps({k: v for k, v in result.items() if k != "missing"}, indent=2))


def _cmd_pairwise_success(args: argparse.Namespace) -> None:
    from cue_training.evaluation.fma.plots.plot_pairwise_success import (
        run_pairwise_success,
    )

    result = run_pairwise_success(
        runs_root=Path(args.runs_root),
        out_dir=Path(args.out_dir),
        stem=args.stem,
    )
    print(json.dumps(result, indent=2))


def _cmd_components(args: argparse.Namespace) -> None:
    from cue_training.evaluation.fma.plots.plot_fidelity_components import (
        run_components,
    )

    result = run_components(
        leaderboard=Path(args.leaderboard),
        distributions=Path(args.distributions),
        out_dir=Path(args.out_dir),
        stem=args.stem,
    )
    print(json.dumps({k: v for k, v in result.items() if k != "notes"}, indent=2))
    notes = result.get("notes") or []
    if notes:
        print(f"Notes ({len(notes)}):")
        for line in notes[:20]:
            print(f"  - {line}")


def _cmd_compare(args: argparse.Namespace) -> None:
    from cue_training.evaluation.fma.compare_fidelity_calibration import run_compare

    result = run_compare(
        leaderboard=Path(args.leaderboard),
        distributions=Path(args.distributions),
        out_dir=Path(args.out_dir),
        stem=args.stem,
        plot=bool(args.plot),
    )
    if result.get("missing"):
        print(f"Notes ({len(result['missing'])}):")
        for line in result["missing"]:
            print(f"  - {line}")
    md = Path(result["correlations_md"])
    if md.is_file():
        print(md.read_text(encoding="utf-8"))
    print(json.dumps({k: v for k, v in result.items() if k != "missing"}, indent=2))


def _cmd_tvd(args: argparse.Namespace) -> None:
    from cue_training.evaluation.fma.tvd import compute_distributions

    report = compute_distributions(
        resolve_run_dir(args.run),
        human_source=args.human_source,
        exclude_labels=_exclude_labels_from_args(args),
        merge_labels=_merge_labels_from_args(args),
    )
    rows = []
    human = report.get("human") or {}
    rows.append(
        {
            "source_id": report.get("human_source") or "human",
            "sim_error_pct": 100.0 * float(human.get("sim_error_rate") or 0.0),
            "tvd_attribution": 0.0,
            "tvd_agent": 0.0,
        }
    )
    for sid, v in (report.get("comparisons") or {}).items():
        rows.append(
            {
                "source_id": sid,
                "sim_error_pct": 100.0 * float(v.get("sim_error_rate") or 0.0),
                "tvd_attribution": v.get("tvd_attribution"),
                "tvd_agent": v.get("tvd_agent"),
            }
        )
    print(
        json.dumps(
            {
                "path": report.get("path")
                or str(resolve_run_dir(args.run) / "distributions.json"),
                "table": rows,
            },
            indent=2,
        )
    )


def _cmd_taxonomy(args: argparse.Namespace) -> None:
    from cue_training.evaluation.fma.review import label_counts, prune_empty_modes
    from cue_training.evaluation.fma.store import load_taxonomy

    run_dir = resolve_run_dir(args.run)
    if args.prune_empty:
        result = prune_empty_modes(run_dir)
        print(
            json.dumps(
                {
                    "removed": result["removed"],
                    "n_remaining": len(result["taxonomy"]),
                    "counts": result["counts"],
                },
                indent=2,
            )
        )
        return
    taxonomy = load_taxonomy(run_dir)
    counts = label_counts(run_dir)
    rows = []
    for mode in taxonomy:
        name = str(mode.get("name") or "")
        rows.append(
            {
                "name": name,
                "n_samples": counts.get(name, 0),
                "description": mode.get("description") or "",
                "n_examples": len(mode.get("examples") or []),
            }
        )
    rows.sort(key=lambda r: (r["n_samples"] == 0, -r["n_samples"], r["name"]))
    empty = [r["name"] for r in rows if r["n_samples"] == 0]
    print(
        json.dumps(
            {
                "n_modes": len(rows),
                "n_empty": len(empty),
                "empty": empty,
                "modes": rows,
            },
            indent=2,
        )
    )


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="cue-fma",
        description="Lightweight Failure Mode Analysis (Appendix D) for tau2.",
    )
    sub = p.add_subparsers(dest="cmd", required=True)

    ing = sub.add_parser("ingest", help="Build corpus.jsonl from human + sim failures")
    ing.add_argument(
        "--out", required=True, help="Run dir or run_id under $CUE_STORAGE_ROOT/fma"
    )
    ing.add_argument("--human", default=None, help="τ-USI normalized.jsonl")
    ing.add_argument(
        "--sim",
        action="append",
        default=[],
        help="Sim rollout JSONL path or glob (repeatable)",
    )
    ing.add_argument(
        "--rewards", default=None, help="Local tau_bench_tasks_unified.json (else HF)"
    )
    ing.add_argument(
        "--tasks-dir", default=None, help="tau2 domains dir with */tasks.json"
    )
    ing.add_argument(
        "--arms", nargs="*", default=None, help="Optional arm filter for sims"
    )
    ing.set_defaults(func=_cmd_ingest)

    bat = sub.add_parser("batch", help="Sample n failures and LLM-propose labels")
    bat.add_argument("--run", required=True)
    bat.add_argument(
        "--batch", type=int, default=None, help="Batch index (default: next)"
    )
    bat.add_argument("--n", type=int, default=100)
    bat.add_argument("--seed", type=int, default=0)
    bat.add_argument(
        "--model", default=None, help=f"LLM model (default {default_model()})"
    )
    bat.add_argument(
        "--workers",
        type=int,
        default=DEFAULT_WORKERS,
        help=f"Concurrent LLM proposes (default {DEFAULT_WORKERS})",
    )
    bat.add_argument("--sample-only", action="store_true")
    bat.set_defaults(func=_cmd_batch)

    disc = sub.add_parser(
        "discover", help="Merge open labels into taxonomy descriptions"
    )
    disc.add_argument("--run", required=True)
    disc.add_argument("--batch", type=int, default=None)
    disc.add_argument("--model", default=None)
    disc.set_defaults(func=_cmd_discover)

    rev = sub.add_parser("review", help="Open local review UI for a batch")
    rev.add_argument("--run", required=True)
    rev.add_argument(
        "--batch",
        type=int,
        default=None,
        help="Batch index (required unless --uncategorized; then defaults to next)",
    )
    rev.add_argument(
        "--uncategorized",
        action="store_true",
        help="Build a review batch from tagged Uncategorized failures and open the UI",
    )
    rev.add_argument(
        "--label",
        default="Uncategorized",
        help="With --uncategorized, which auto-tag label to review (default Uncategorized)",
    )
    rev.add_argument(
        "--n",
        type=int,
        default=None,
        help="With --uncategorized, cap how many to review this pass (default: all remaining)",
    )
    rev.add_argument(
        "--seed", type=int, default=0, help="Shuffle seed for --uncategorized"
    )
    rev.add_argument("--host", default="127.0.0.1")
    rev.add_argument("--port", type=int, default=8765)
    rev.set_defaults(func=_cmd_review)

    tag = sub.add_parser("tag", help="Classify remaining failures with frozen taxonomy")
    tag.add_argument("--run", required=True)
    tag.add_argument("--model", default=None)
    tag.add_argument(
        "--workers",
        type=int,
        default=DEFAULT_WORKERS,
        help=f"Concurrent LLM classify calls (default {DEFAULT_WORKERS})",
    )
    tag.add_argument(
        "--force",
        action="store_true",
        help="Reclassify everything (ignore prior tagged.jsonl; review decisions still apply unless also --relabel)",
    )
    tag.add_argument(
        "--relabel",
        action="append",
        default=[],
        help="Drop prior tags with this label and reclassify those rows only (repeatable). "
        'Example: --relabel "Critical User Simulator Error"',
    )
    tag.add_argument(
        "--reasoning-effort",
        default="low",
        help="Reasoning budget for reasoning models (low keeps closed-set tagging fast; "
        "empty string uses the model default)",
    )
    tag.set_defaults(func=_cmd_tag)

    tvd = sub.add_parser(
        "tvd",
        help="Sim error %% + attribution TVD (user/env/agent) + agent-failure TVD vs human",
    )
    tvd.add_argument("--run", required=True)
    tvd.add_argument("--human-source", default="human")
    tvd.add_argument(
        "--exclude-label",
        action="append",
        default=[],
        help="Drop this failure mode before computing TVD (repeatable)",
    )
    tvd.add_argument(
        "--no-sim-error",
        action="store_true",
        help="Exclude Premature User Stop, User Identity / Task Derailment, "
        "Other Simulator Error (and legacy Critical User Simulator Error)",
    )
    tvd.add_argument(
        "--merge-label",
        action="append",
        default=[],
        help="Remap OLD=NEW before TVD (repeatable), e.g. 'User Data Leakage=User Identity / Task Derailment'",
    )
    tvd.add_argument(
        "--user-error",
        action="store_true",
        help="Merge Early Stop, Other Sim Error, Data Leakage, and Identity Derail "
        "into one User Error label before TVD",
    )
    tvd.add_argument(
        "--leakage-as-sim-error",
        action="store_true",
        help='Shorthand for --merge-label "User Data Leakage=User Identity / Task Derailment"',
    )
    tvd.set_defaults(func=_cmd_tvd)

    plot = sub.add_parser(
        "plot",
        help="Write percent-combined failure-mix figure from tagged.jsonl",
    )
    plot.add_argument("--run", required=True)
    plot.add_argument(
        "--out-dir",
        default=None,
        help="Figure/CSV output dir (default: <run>/figures)",
    )
    plot.add_argument(
        "--exclude-label",
        action="append",
        default=[],
        help="Drop this failure mode before plotting (repeatable)",
    )
    plot.add_argument(
        "--no-sim-error",
        action="store_true",
        help="Exclude Premature User Stop, User Identity / Task Derailment, "
        "Other Simulator Error (and legacy Critical User Simulator Error)",
    )
    plot.add_argument(
        "--merge-label",
        action="append",
        default=[],
        help="Remap OLD=NEW before plotting (repeatable)",
    )
    plot.add_argument(
        "--user-error",
        action="store_true",
        help="Merge Early Stop, Other Sim Error, Data Leakage, and Identity Derail "
        "into one User Error bar (Critical User Simulator Error)",
    )
    plot.add_argument(
        "--leakage-as-sim-error",
        action="store_true",
        help='Shorthand for --merge-label "User Data Leakage=User Identity / Task Derailment"',
    )
    plot.set_defaults(func=_cmd_plot)

    scatter = sub.add_parser(
        "scatter",
        help="Six-panel naturalness/mimicry/coverage × ΔSuc/TVD-agent scatters",
    )
    scatter.add_argument(
        "--leaderboard",
        default="extended-output/metrics/tau2_customer-service/leaderboard.json",
        help="Aggregate leaderboard JSON (uses domain=macro rows)",
    )
    scatter.add_argument(
        "--distributions",
        default="fma/distributions.json",
        help="FMA distributions.json with tvd_agent per source_id",
    )
    scatter.add_argument("--out-dir", default="fma/figures")
    scatter.add_argument("--stem", default="metrics_vs_delta_suc_tvd")
    scatter.add_argument("--no-human", action="store_true")
    scatter.add_argument("--no-plot", action="store_true")
    scatter.set_defaults(func=_cmd_scatter)

    compare = sub.add_parser(
        "compare",
        help=(
            "Pearson table: Nat-S2R/TT, Mim-AVA/PT3, Cov-S2R/SD Chamfer vs |ΔSuc|, User Error %%, TVD_Agent"
        ),
    )
    compare.add_argument(
        "--leaderboard",
        default="output/metrics/tau2_customer-service/leaderboard.json",
        help="Aggregate leaderboard JSON (uses domain=macro rows)",
    )
    compare.add_argument(
        "--distributions",
        default="fma/distributions.json",
        help="FMA distributions.json with sim_error_rate + tvd_agent per source_id",
    )
    compare.add_argument("--out-dir", default="fma/figures")
    compare.add_argument("--stem", default="fidelity_vs_calibration")
    compare.add_argument(
        "--plot",
        action="store_true",
        help="Also write a 6×3 scatter PNG/PDF",
    )
    compare.set_defaults(func=_cmd_compare)

    components = sub.add_parser(
        "components",
        help="τ² dot chart: 3 fidelity + 3 calibration components, right = better",
    )
    components.add_argument(
        "--leaderboard",
        default="output/metrics/tau2_customer-service/leaderboard.json",
        help="Aggregate leaderboard JSON (uses domain=macro rows)",
    )
    components.add_argument(
        "--distributions",
        default="fma/distributions.json",
        help="FMA distributions.json with sim_error_rate + tvd_agent",
    )
    components.add_argument("--out-dir", default="fma/figures")
    components.add_argument("--stem", default="fidelity_calibration_components")
    components.set_defaults(func=_cmd_components)

    pairwise = sub.add_parser(
        "pairwise-success",
        help="τ² compositional bars: both fail / both succeed / human-only / sim-only",
    )
    pairwise.add_argument(
        "--runs-root",
        default="output/metrics/tau2_customer-service/_runs",
        help="Metrics run tree with per-seed summary.json files",
    )
    pairwise.add_argument("--out-dir", default="fma/figures")
    pairwise.add_argument("--stem", default="pairwise_success_composition")
    pairwise.set_defaults(func=_cmd_pairwise_success)

    tax = sub.add_parser("taxonomy", help="List taxonomy modes with sample counts")
    tax.add_argument("--run", required=True)
    tax.add_argument(
        "--prune-empty",
        action="store_true",
        help="Drop modes with 0 proposal/decision/tagged samples (keeps Uncategorized)",
    )
    tax.set_defaults(func=_cmd_taxonomy)

    return p


def main(argv: list[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.cmd in {
        "ingest",
        "batch",
        "review",
        "tag",
        "tvd",
        "discover",
        "taxonomy",
        "plot",
        "scatter",
        "compare",
        "components",
        "pairwise-success",
    }:
        # Ensure default root exists for run_id style paths.
        default_fma_root().mkdir(parents=True, exist_ok=True)
    args.func(args)


if __name__ == "__main__":
    main()
