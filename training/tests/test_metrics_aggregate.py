"""Tests for aggregate cross-simulator metrics fit + leaderboard + seed pooling."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from cue_training.evaluation.metrics import cache
from cue_training.evaluation.metrics.aggregate import (
    build_leaderboard,
    discover_candidates,
    macro_across_domains,
    method_family,
    pool_across_seeds,
    resolve_base_paths,
    summaries_to_rows,
)
from cue_training.evaluation.metrics.data import load_baseline_paths
from cue_training.evaluation.metrics.stats import Aggregate


def _write_rollout(path: Path, *, episode_id: str, human: str, rollout: str, sim_tag: str = "") -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    rec = {
        "episode_id": episode_id,
        "real_conversation": [
            {"role": "user", "content": human},
            {"role": "assistant", "content": "ok"},
        ],
        "rollout_conversation": [
            {"role": "user", "content": rollout},
            {"role": "assistant", "content": "ok"},
        ],
        "metadata": {"arm": "as_is", "domain": "writing", "sim": sim_tag},
    }
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(rec) + "\n")


def test_load_baseline_paths_one_sim_per_episode(tmp_path: Path):
    gpt = tmp_path / "gpt" / "rollout.simulatorarena.jsonl"
    llama = tmp_path / "llama" / "rollout.simulatorarena.jsonl"
    _write_rollout(gpt, episode_id="e1", human="please help", rollout="gpt user text")
    _write_rollout(llama, episode_id="e1", human="please help", rollout="llama user text")
    _write_rollout(gpt, episode_id="e2", human="second", rollout="gpt two")
    _write_rollout(llama, episode_id="e2", human="second", rollout="llama two")

    baseline = load_baseline_paths([gpt, llama], sim_labels=["gpt", "llama"], seed=0)
    assert len(baseline) == 2
    source_ids = {ep.metadata["aggregate_source_episode_id"] for ep in baseline.values()}
    assert source_ids == {"e1", "e2"}
    assert all(ep.metadata["aggregate_sim"] in {"gpt", "llama"} for ep in baseline.values())
    again = load_baseline_paths([gpt, llama], sim_labels=["gpt", "llama"], seed=0)
    assert {k: again[k].metadata["aggregate_sim"] for k in again} == {
        k: baseline[k].metadata["aggregate_sim"] for k in baseline
    }
    assert len(load_baseline_paths([gpt, llama], sim_labels=["gpt", "llama"], seed=1)) == 2
    for ep in baseline.values():
        sim = ep.metadata["aggregate_sim"]
        eid = ep.metadata["aggregate_source_episode_id"]
        if eid == "e1":
            assert ep.human[0]["content"] == "please help"
            assert ep.base[0]["content"] == ("gpt user text" if sim == "gpt" else "llama user text")
        else:
            assert ep.human[0]["content"] == "second"
            assert ep.base[0]["content"] == ("gpt two" if sim == "gpt" else "llama two")


def test_paths_hash_and_aggregate_manifest(tmp_path: Path):
    a = tmp_path / "a.jsonl"
    b = tmp_path / "b.jsonl"
    a.write_text("{}\n", encoding="utf-8")
    b.write_text("{}\n", encoding="utf-8")
    state = tmp_path / "state"
    cfg = {"version": 9, "metrics": ["classifier/sim2real"]}
    cache.write_manifest(state, baselines=[a, b], config=cfg)
    assert cache.manifest_matches(state, baselines=[a, b], config=cfg)
    assert cache.manifest_matches(state, baselines=[b, a], config=cfg)
    assert not cache.manifest_matches(state, baseline=a, config=cfg)
    b.write_text('{"x":1}\n', encoding="utf-8")
    assert not cache.manifest_matches(state, baselines=[a, b], config=cfg)


def _complete_job(
    root: Path,
    *,
    run_id: str,
    benchmark: str,
    domain: str,
    kind: str,
    method: str,
    sim: str,
) -> Path:
    job = root / run_id / benchmark / domain / kind / method / sim
    job.mkdir(parents=True, exist_ok=True)
    (job / "COMPLETE").write_text("ok\n", encoding="utf-8")
    rollout = job / "rollout.simulatorarena.jsonl"
    _write_rollout(rollout, episode_id="e1", human="h", rollout=f"{sim} text", sim_tag=sim)
    return rollout


def test_discover_candidates_and_resolve_base(tmp_path: Path):
    root = tmp_path / "rollouts"
    _complete_job(
        root, run_id="base", benchmark="simulatorarena", domain="writing",
        kind="baseline", method="base", sim="gpt",
    )
    _complete_job(
        root, run_id="base", benchmark="simulatorarena", domain="writing",
        kind="baseline", method="base", sim="llama",
    )
    _complete_job(
        root, run_id="cue-general", benchmark="simulatorarena", domain="writing",
        kind="cue", method="general", sim="gpt",
    )
    _complete_job(
        root, run_id="realusersim", benchmark="simulatorarena", domain="writing",
        kind="baseline", method="realusersim", sim="llama",
    )
    _complete_job(
        root, run_id="cue-general", benchmark="simulatorarena", domain="math",
        kind="cue", method="general", sim="gpt",
    )

    bases = resolve_base_paths(
        root, benchmark="simulatorarena", domain="writing", simulators=["gpt", "llama", "gemini"]
    )
    assert [s for s, _ in bases] == ["gpt", "llama"]

    jobs = discover_candidates(root, benchmark="simulatorarena", domain="writing")
    labels = {(j.run_id, j.simulator) for j in jobs}
    assert ("base", "gpt") in labels
    assert ("cue-general", "gpt") in labels
    assert ("realusersim", "llama") in labels
    assert all(j.domain == "writing" for j in jobs)

    filtered = discover_candidates(
        root, benchmark="simulatorarena", domain="writing", methods={"realusersim"}
    )
    assert {(j.run_id, j.simulator) for j in filtered} == {("realusersim", "llama")}


def test_discover_seed_layout_and_family(tmp_path: Path):
    root = tmp_path / "rollouts"
    for seed in (0, 1, 2):
        seed_root = root / f"seed-{seed}"
        _complete_job(
            seed_root, run_id="baseline-base-once", benchmark="simulatorarena", domain="writing",
            kind="baseline", method="base", sim="gpt",
        )
        _complete_job(
            seed_root, run_id="baseline-base-once", benchmark="simulatorarena", domain="writing",
            kind="baseline", method="base", sim="llama",
        )
        _complete_job(
            seed_root, run_id=f"cue-general-seed-{seed}", benchmark="simulatorarena", domain="writing",
            kind="cue", method="general", sim="gpt",
        )
        _complete_job(
            seed_root, run_id=f"baseline-seed-{seed}", benchmark="simulatorarena", domain="writing",
            kind="baseline", method="realusersim", sim="gpt",
        )

    bases = resolve_base_paths(
        root, benchmark="simulatorarena", domain="writing",
        base_run_id="baseline-base-once", simulators=["gpt", "llama"], seeds=[0, 1, 2],
    )
    assert [s for s, _ in bases] == ["gpt", "llama"]

    jobs = discover_candidates(
        root, benchmark="simulatorarena", domain="writing", seeds=[0, 1, 2]
    )
    cue = [j for j in jobs if j.kind == "cue"]
    assert len(cue) == 3
    assert {j.family for j in cue} == {"cue-general"}
    assert {j.seed for j in cue} == {0, 1, 2}
    rus = [j for j in jobs if j.method == "realusersim"]
    assert len(rus) == 3
    assert {j.family for j in rus} == {"realusersim"}


def test_method_family_strips_seed():
    assert method_family("cue-general-seed-2", kind="cue", method="general") == "cue-general"
    assert method_family("baseline-seed-0", kind="baseline", method="usp") == "usp"
    assert (
        method_family(
            "hosted_baselines_multiseed-seed-1", kind="baseline", method="userlm"
        )
        == "userlm"
    )
    assert (
        method_family(
            "hosted_baselines_multiseed-seed-0", kind="baseline", method="base"
        )
        == "base"
    )
    assert method_family("baseline-base-once", kind="baseline", method="base") == "base"
    assert method_family("baseline-seed-2", kind="baseline", method="base") == "base"
    assert method_family("human", kind="human", method="human") == "human"


def test_resolve_base_paths_seeded_multiseed(tmp_path: Path):
    """Floor-base can live inside a per-seed multiseed run (not only base-once)."""

    root = tmp_path / "rollouts"
    for seed in (0, 1, 2):
        seed_root = root / f"seed-{seed}"
        run_id = f"hosted_baselines_multiseed-seed-{seed}"
        for sim in ("gpt", "llama", "gemini"):
            _complete_job(
                seed_root,
                run_id=run_id,
                benchmark="simulatorarena",
                domain="writing",
                kind="baseline",
                method="base",
                sim=sim,
            )
        _complete_job(
            seed_root,
            run_id=run_id,
            benchmark="simulatorarena",
            domain="writing",
            kind="baseline",
            method="usp",
            sim="llama",
        )

    # Explicit multiseed prefix (resolves to -seed-0 under seed-0/).
    bases = resolve_base_paths(
        root,
        benchmark="simulatorarena",
        domain="writing",
        base_run_id="hosted_baselines_multiseed",
        simulators=["gpt", "llama", "gemini"],
        seeds=[0, 1, 2],
    )
    assert [s for s, _ in bases] == ["gpt", "llama", "gemini"]
    assert "hosted_baselines_multiseed-seed-0" in str(bases[0][1])

    # Legacy base-once id still falls back to the multiseed base layout.
    bases_fallback = resolve_base_paths(
        root,
        benchmark="simulatorarena",
        domain="writing",
        base_run_id="baseline-base-once",
        simulators=["gpt", "llama"],
        seeds=[0],
    )
    assert [s for s, _ in bases_fallback] == ["gpt", "llama"]

    jobs = discover_candidates(
        root, benchmark="simulatorarena", domain="writing", seeds=[0, 1, 2]
    )
    base_jobs = [j for j in jobs if j.method == "base"]
    assert len(base_jobs) == 9  # 3 seeds × 3 sims
    assert {j.family for j in base_jobs} == {"base"}
    assert {j.seed for j in base_jobs} == {0, 1, 2}


def test_write_human_proxy_rollout(tmp_path: Path):
    from cue_training.evaluation.metrics.aggregate import write_human_proxy_rollout
    from cue_training.evaluation.metrics.data import load_episodes, load_baseline

    src = tmp_path / "base" / "rollout.simulatorarena.jsonl"
    _write_rollout(src, episode_id="e1", human="real user text", rollout="sim text")
    out = tmp_path / "human.jsonl"
    write_human_proxy_rollout(src, out)
    rows = [json.loads(line) for line in out.read_text().splitlines() if line.strip()]
    assert len(rows) == 1
    assert rows[0]["metadata"]["arm"] == "human"
    assert rows[0]["rollout_conversation"] == rows[0]["real_conversation"]
    assert rows[0]["rollout_conversation"][0]["content"] == "real user text"

    baseline = load_baseline(src)
    episodes, _ = load_episodes(src, out)
    assert len(episodes) == 1
    assert episodes[0].proxy == episodes[0].human == baseline["e1"].human
    assert episodes[0].arm == "human"


def test_pool_across_seeds_and_domain_macro():
    metric = "classifier/sim2real"
    summaries = []
    for seed, writing, math in ((0, 0.4, 0.6), (1, 0.5, 0.7), (2, 0.6, 0.8)):
        for domain, mean in (("writing", writing), ("math", math)):
            summaries.append({
                "candidate_meta": {
                    "run_id": f"cue-general-seed-{seed}",
                    "family": "cue-general",
                    "method": "general",
                    "kind": "cue",
                    "simulator": "gpt",
                    "seed": seed,
                    "domain": domain,
                },
                "group_meta": {"paired": {"arm": "paired", "domain": domain}},
                "aggregates_by_group": {
                    "paired": [Aggregate(metric, mean, sample_size=10).to_json()]
                },
            })
    seed_rows = summaries_to_rows(summaries, metric_names=[metric])
    pooled = pool_across_seeds(seed_rows, metric_names=[metric])
    by_dom = {r["domain"]: r for r in pooled if r["family"] == "cue-general"}
    assert abs(by_dom["writing"]["metrics"][metric]["mean"] - 0.5) < 1e-9
    assert abs(by_dom["math"]["metrics"][metric]["mean"] - 0.7) < 1e-9
    # ± is sample stdev of seed means {0.4, 0.5, 0.6} → 0.1 (not SEM CI).
    assert by_dom["writing"]["metrics"][metric]["confidence_interval"] == pytest.approx(0.1)
    assert by_dom["writing"]["metrics"][metric]["standard_deviation"] == pytest.approx(0.1)
    assert by_dom["writing"]["n_seeds"] == 3

    macros = macro_across_domains(
        seed_rows, metric_names=[metric], domains=["writing", "math"]
    )
    assert len(macros) == 1
    # Per-seed macros: 0.5, 0.6, 0.7 → mean 0.6, stdev 0.1
    assert abs(macros[0]["metrics"][metric]["mean"] - 0.6) < 1e-9
    assert macros[0]["metrics"][metric]["confidence_interval"] == pytest.approx(0.1)
    assert macros[0]["domain"] == "macro"
    assert set(macros[0]["domains_averaged"]) == {"writing", "math"}

    rows, md = build_leaderboard(pooled + macros, metric_names=[metric])
    assert any(r["domain"] == "macro" for r in rows)
    assert "Seeds" in md


def test_macro_ignores_arm_pooled_rows_that_steal_a_subdomain():
    """An arm-pooled row with a buggy domain=airline stamp must not erase airline success."""

    success = "env/tau2_success_rate"
    f1 = "env/tau2_task_success"
    pooled = "coverage/styledistance_behavioral"

    def _f1(rate: float) -> dict:
        return Aggregate(
            f1,
            0.5,
            sample_size=100,
            extras={"env_success_rate": {"mean": rate, "n": 100}},
        ).to_json()

    summaries = [
        {
            "candidate_meta": {
                "run_id": "base-seed-0",
                "family": "base",
                "method": "base",
                "kind": "baseline",
                "simulator": "gemini",
                "seed": 0,
                "domain": "customer-service",
            },
            "group_meta": {
                "as_is/airline": {"arm": "as_is", "domain": "airline"},
                "as_is/retail": {"arm": "as_is", "domain": "retail"},
                "as_is": {"arm": "as_is", "domain": "airline"},  # the bug stamp
            },
            "aggregates_by_group": {
                "as_is/airline": [_f1(0.80)],
                "as_is/retail": [_f1(0.60)],
                "as_is": [Aggregate(pooled, 0.02, sample_size=300).to_json()],
            },
        }
    ]
    seed_rows = summaries_to_rows(summaries, metric_names=[success, f1, pooled])
    macros = macro_across_domains(
        seed_rows, metric_names=[success, f1, pooled], domains=["customer-service"]
    )
    assert len(macros) == 1
    # Equal-weight mean of airline 0.80 and retail 0.60 — not a retail copy.
    assert macros[0]["metrics"][success]["mean"] == pytest.approx(0.70)
    assert set(macros[0]["domains_averaged"]) == {"airline", "retail"}


def test_build_leaderboard_puts_human_first():
    summaries = [
        {
            "candidate_meta": {
                "run_id": "cue-general", "family": "cue-general",
                "method": "general", "kind": "cue", "simulator": "gpt", "seed": 0,
            },
            "group_meta": {"paired": {"arm": "paired", "domain": "writing"}},
            "aggregates_by_group": {
                "paired": [Aggregate("classifier/sim2real", 0.5, confidence_interval=0.1, sample_size=10).to_json()]
            },
        },
        {
            "candidate_meta": {
                "run_id": "human", "family": "human", "method": "human",
                "kind": "human", "simulator": "—", "upper_bound": True, "seed": None,
            },
            "group_meta": {"human": {"arm": "human", "domain": "writing"}},
            "aggregates_by_group": {
                "human": [Aggregate("classifier/sim2real", 0.9, confidence_interval=0.05, sample_size=10).to_json()]
            },
        },
    ]
    seed_rows = summaries_to_rows(summaries, metric_names=["classifier/sim2real"])
    pooled = pool_across_seeds(seed_rows, metric_names=["classifier/sim2real"])
    rows, md = build_leaderboard(pooled, metric_names=["classifier/sim2real"])
    assert rows[0]["run_id"] == "human"
    assert "upper bound" in md.lower()
    assert md.index("| human |") < md.index("| cue-general |")


def test_leaderboard_includes_tau2_success_rate():
    from cue_training.evaluation.metrics.aggregate import leaderboard_metric_names

    metric = "env/tau2_task_success"
    lb = leaderboard_metric_names([metric, "classifier/sim2real"])
    assert lb == ["env/tau2_success_rate", metric, "classifier/sim2real"]

    summaries = []
    for seed, rate, f1 in ((0, 0.4, 0.5), (1, 0.6, 0.7)):
        summaries.append({
            "candidate_meta": {
                "run_id": f"cue-general-seed-{seed}",
                "family": "cue-general",
                "method": "general",
                "kind": "cue",
                "simulator": "gpt",
                "seed": seed,
                "domain": "airline",
            },
            "group_meta": {"paired": {"arm": "paired", "domain": "airline"}},
            "aggregates_by_group": {
                "paired": [
                    Aggregate(
                        metric,
                        f1,
                        sample_size=10,
                        extras={"env_success_rate": {"mean": rate, "ci": 0.05, "n": 10}},
                    ).to_json()
                ]
            },
        })
    seed_rows = summaries_to_rows(summaries, metric_names=lb)
    assert seed_rows[0]["seed_means"]["env/tau2_success_rate"] == pytest.approx(0.4)
    pooled = pool_across_seeds(seed_rows, metric_names=lb)
    assert abs(pooled[0]["metrics"]["env/tau2_success_rate"]["mean"] - 0.5) < 1e-9
    _, md = build_leaderboard(pooled, metric_names=lb)
    assert "Success rate" in md
    assert md.index("Success rate") < md.index("Success F1")


def test_leaderboard_splits_metric_families():
    names = [
        "env/tau2_task_success",
        "classifier/sim2real",
        "judge/turing_sonnet_qwen",
        "mimicry/wegmann_ava",
        "mimicry/paired_audit",
        "coverage/styledistance_behavioral",
    ]
    row = {
        "run_id": "cue-general",
        "family": "cue-general",
        "simulator": "gpt",
        "arm": "paired",
        "domain": "airline",
        "n_seeds": 1,
        "metrics": {
            name: Aggregate(name, 0.5, sample_size=2).to_json() for name in names
        },
    }
    _, md = build_leaderboard([row], metric_names=names)

    assert [line for line in md.splitlines() if line.startswith("## ")] == [
        "## Task",
        "## Naturalness",
        "## Mimicry",
        "## Coverage",
    ]
    task, naturalness, mimicry, coverage = (
        md.split("## Task", 1)[1]
        .split("## Naturalness", 1)[0],
        md.split("## Naturalness", 1)[1].split("## Mimicry", 1)[0],
        md.split("## Mimicry", 1)[1].split("## Coverage", 1)[0],
        md.split("## Coverage", 1)[1],
    )
    assert "Success F1" in task and "Sim2Real P(human)" not in task
    assert "Sim2Real P(human)" in naturalness and "Turing |0.5-mean_P|" in naturalness
    assert "Wegmann AVA" in mimicry and "Paired audit" in mimicry
    assert "SD coverage" in coverage and "Wegmann AVA" not in coverage


def test_run_aggregate_smoke(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """End-to-end wiring with fit/score stubbed — verifies discovery → pooled leaderboard."""

    from cue_training.evaluation.metrics import aggregate as agg

    root = tmp_path / "rollouts"
    out = tmp_path / "metrics" / "aggregate"
    for sim in ("gpt", "llama"):
        _complete_job(
            root, run_id="base", benchmark="simulatorarena", domain="writing",
            kind="baseline", method="base", sim=sim,
        )
    _complete_job(
        root, run_id="cue-general", benchmark="simulatorarena", domain="writing",
        kind="cue", method="general", sim="gpt",
    )

    fitted: list = []

    def fake_fit(paths, cache_dir, **kwargs):
        fitted.append((list(paths), Path(cache_dir)))
        Path(cache_dir).mkdir(parents=True, exist_ok=True)
        cache.write_manifest(
            Path(cache_dir),
            baselines=list(paths),
            config={"version": 9, "metrics": ["classifier/sim2real"]},
        )
        return {}

    def fake_run_one(baseline_path, candidate_path, **kwargs):
        run_dir = Path(kwargs["run_dir"])
        run_dir.mkdir(parents=True, exist_ok=True)
        parts = run_dir.parts
        is_human = "human" in parts
        mean = 0.9 if is_human else (0.3 if run_dir.name == "gpt" else 0.4)
        arm = "human" if is_human else "as_is"
        merge = bool(kwargs.get("merge"))
        metrics = list(kwargs.get("metrics") or ["classifier/sim2real"])
        existing = {}
        summary_path = run_dir / "summary.json"
        if merge and summary_path.is_file():
            existing = {
                a["metric_name"]: a
                for a in (json.loads(summary_path.read_text())
                          .get("aggregates_by_group", {})
                          .get(arm) or [])
            }
        for name in metrics:
            existing[name] = Aggregate(
                name,
                mean,
                confidence_interval=0.1,
                sample_size=5,
            ).to_json()
        payload = {
            "aggregates_by_group": {arm: list(existing.values())},
            "group_meta": {arm: {"arm": arm, "domain": "writing"}},
            "telemetry_stats": {"episodes_by_group": {arm: 5}},
        }
        summary_path.write_text(json.dumps(payload), encoding="utf-8")
        (run_dir / "summary.md").write_text("# ok\n", encoding="utf-8")

    monkeypatch.setattr(agg, "fit_aggregate_state", fake_fit)
    monkeypatch.setattr(agg, "run_one", fake_run_one)

    md_path = agg.run_aggregate(
        benchmark="simulatorarena",
        domains=["writing"],
        rollouts_root=root,
        out_dir=out,
        metrics=["classifier/sim2real"],
        simulators=["gpt", "llama"],
        seeds=None,
    )
    assert md_path.is_file()
    data = json.loads((out / "simulatorarena_writing" / "leaderboard.json").read_text())
    assert data["includes_human_upper_bound"] is True
    assert data["rows"][0]["run_id"] == "human"
    assert fitted and len(fitted[0][0]) == 2
    assert "pooling" in data

    # Selective re-run with --merge keeps prior columns and adds the new one.
    seen_merge: list[bool] = []

    def fake_run_one_merge(baseline_path, candidate_path, **kwargs):
        seen_merge.append(bool(kwargs.get("merge")))
        return fake_run_one(baseline_path, candidate_path, **kwargs)

    monkeypatch.setattr(agg, "run_one", fake_run_one_merge)
    agg.run_aggregate(
        benchmark="simulatorarena",
        domains=["writing"],
        rollouts_root=root,
        out_dir=out,
        metrics=["env/tau2_task_success"],
        simulators=["gpt", "llama"],
        seeds=None,
        merge=True,
    )
    assert seen_merge and all(seen_merge)
    present = agg.metrics_present_in_summaries(
        [
            json.loads(p.read_text())
            for p in (out / "simulatorarena_writing" / "_runs").rglob("summary.json")
        ]
    )
    assert "classifier/sim2real" in present
    assert "env/tau2_task_success" in present


def test_metrics_present_in_summaries_orders_defaults_first():
    from cue_training.evaluation.metrics import aggregate as agg

    summaries = [
        {
            "aggregates_by_group": {
                "paired": [
                    {"metric_name": "zzz/extra", "mean": 1.0},
                    {"metric_name": "classifier/sim2real", "mean": 0.5},
                    {"metric_name": "env/tau2_task_success", "mean": 0.7},
                ]
            }
        }
    ]
    names = agg.metrics_present_in_summaries(summaries)
    assert names.index("env/tau2_task_success") < names.index("classifier/sim2real")
    assert names[-1] == "zzz/extra"
