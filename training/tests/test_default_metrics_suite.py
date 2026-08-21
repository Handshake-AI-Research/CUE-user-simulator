"""Focused tests for the default metrics suite redesign."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest

from cue_training.evaluation.metrics.data import BaselineEpisode, Episode
from cue_training.evaluation.metrics.metrics import api as api_metrics
from cue_training.evaluation.metrics.metrics import authenticity as authenticity_metrics
from cue_training.evaluation.metrics.metrics import coverage as C
from cue_training.evaluation.metrics.policy import skip_ava_mrr, skip_paired_audit
from cue_training.evaluation.metrics.run import ALL_METRICS, DEFAULT_METRICS, _parse_metrics


def _ep(
    eid: str,
    *,
    arm: str = "paired",
    domain: str = "airline",
    human: list[str] | None = None,
    proxy: list[str] | None = None,
) -> Episode:
    human = human or [f"human-{eid}-a", f"human-{eid}-b"]
    proxy = proxy or [f"proxy-{eid}-a", f"proxy-{eid}-b"]
    return Episode(
        episode_id=eid,
        human=[{"role": "user", "content": t} for t in human]
        + [{"role": "assistant", "content": "ok"}],
        base=[{"role": "user", "content": "base"}],
        proxy=[{"role": "user", "content": t} for t in proxy]
        + [{"role": "assistant", "content": "ok"}],
        metadata={"arm": arm, "domain": domain},
        base_metadata={"domain": domain},
        arm=arm,
        domain=domain,
    )


def test_default_metrics_suite():
    assert DEFAULT_METRICS == [
        "env/tau2_success_rate",
        "env/tau2_task_success",
        "classifier/sim2real",
        "judge/turing_sonnet_qwen",
        "mimicry/wegmann_ava",
        "mimicry/paired_audit",
        "coverage/sim2real_behavioral",
        "coverage/styledistance_behavioral",
    ]
    assert _parse_metrics("default") == DEFAULT_METRICS
    assert _parse_metrics("all") == DEFAULT_METRICS
    for name in DEFAULT_METRICS:
        assert name in ALL_METRICS


def test_unsupported_metrics_are_rejected():
    with pytest.raises(ValueError, match="unsupported metric"):
        _parse_metrics("coverage/styledistance_mauve")


def test_skip_matrix():
    assert skip_ava_mrr(kind="baseline", method="realusersim", arm="as_is")
    assert skip_ava_mrr(kind="baseline", method="userlm", arm="as_is")
    assert skip_ava_mrr(kind="baseline", method="usp", arm="sample_diverse")
    assert not skip_ava_mrr(kind="baseline", method="realusersim_paired_noex", arm="paired")
    assert not skip_ava_mrr(kind="baseline", method="ppol", arm="paired")

    assert skip_paired_audit(kind="baseline", method="usp", arm="sample_diverse")
    assert skip_paired_audit(kind="cue", method="general", arm="as_is")
    assert not skip_paired_audit(kind="baseline", method="realusersim", arm="as_is")
    assert not skip_paired_audit(kind="baseline", method="realusersim_paired_noex", arm="paired")


def test_strip_terminations():
    text = "hello ###STOP### world END OF CONVERSATION"
    assert "STOP" not in api_metrics._strip_terminations(text)
    assert "hello" in api_metrics._strip_terminations(text)


def test_completion_kwargs_temperature_one_for_claude_sonnet_5():
    kw = api_metrics._completion_kwargs("anthropic/claude-sonnet-5", "pick A or B")
    assert kw["temperature"] == 1
    kw0 = api_metrics._completion_kwargs("anthropic/claude-sonnet-4-5", "pick A or B")
    assert kw0["temperature"] == 0


def test_default_turing_judges_claude_only():
    judges = api_metrics._default_turing_judges()
    assert len(judges) == 1
    assert judges[0]["name"] == "claude-sonnet-5"
    assert "claude" in judges[0]["model"]


def test_completion_kwargs_omits_reasoning_effort_for_openrouter():
    kw = api_metrics._completion_kwargs(
        "openrouter/qwen/qwen3.8-2.4t-a95b", "pick A or B"
    )
    assert "reasoning_effort" not in kw
    assert kw.get("chat_template_kwargs", {}).get("enable_thinking") is False
    kw_claude = api_metrics._completion_kwargs("anthropic/claude-sonnet-5", "pick A or B")
    assert kw_claude.get("reasoning_effort") == "none"


def test_turing_qwen_judge_local_vllm(monkeypatch):
    monkeypatch.setenv("MIRROR_TURING_QWEN_API_BASE", "http://127.0.0.1:8000/v1")
    monkeypatch.setenv("MIRROR_TURING_QWEN", "Qwen/Qwen3-32B")
    judge = api_metrics._turing_qwen_judge()
    assert judge["model"] == "hosted_vllm/Qwen/Qwen3-32B"
    assert judge["key"] == "HOSTED_VLLM_API_KEY"
    assert api_metrics._available_judges([judge]) == [judge]
    kw = api_metrics._completion_kwargs(judge["model"], "pick A or B")
    assert kw["api_base"] == "http://127.0.0.1:8000/v1"
    assert "reasoning_effort" not in kw
    assert kw.get("extra_body", {}).get("chat_template_kwargs", {}).get("enable_thinking") is False


def test_turing_cache_key_includes_proxy_content(tmp_path: Path):
    """Different proxies for the same episode id must not share a judgment cache entry."""

    from cue_training.evaluation.metrics import cache

    baseline = {
        f"e{i}": BaselineEpisode(
            episode_id=f"e{i}",
            human=[{"role": "user", "content": f"human text {i} " * 20}],
            base=[{"role": "user", "content": f"base text {i} " * 20}],
            metadata={"domain": "airline"},
            raw={},
        )
        for i in range(12)
    }
    api_metrics.fit_turing_state(baseline, tmp_path, n_eval=4, n_fewshot=2, seed=0)
    eval_ids = cache.load_pickle(tmp_path, "turing_eval_ids.pkl")["airline"]
    eid = eval_ids[0]

    calls: list[str] = []

    def _fake_batch(reqs, *, concurrency=None):
        out = []
        for _model, prompt, _mt in reqs:
            calls.append(prompt)
            a_block = prompt.split("A:\n", 1)[-1].split("\n\nB:", 1)[0]
            out.append("A" if "human text" in a_block else "B")
        return out

    judges = [{"name": "j", "model": "m", "key": "HOSTED_VLLM_API_KEY"}]
    eps_a = [_ep(eid, arm="paired", domain="airline", proxy=["proxy-AAA " * 20])]
    eps_b = [_ep(eid, arm="paired", domain="airline", proxy=["proxy-BBB " * 20])]
    # Keep human text matching baseline so strip/user_text still works via episode.
    for ep in (*eps_a, *eps_b):
        ep.human = [{"role": "user", "content": f"human text {eid.split('e')[-1] if False else eval_ids.index(eid)} " * 20}]
    # Simpler: set human from baseline id index
    idx = int(eid[1:]) if eid[1:].isdigit() else 0
    for ep in (*eps_a, *eps_b):
        ep.human = [{"role": "user", "content": f"human text {idx} " * 20}]

    with patch.object(api_metrics, "completion_batch", side_effect=_fake_batch):
        with patch.object(api_metrics, "_available_judges", return_value=judges):
            api_metrics.turing_sonnet_qwen(eps_a, tmp_path)
            n_first = len(calls)
            api_metrics.turing_sonnet_qwen(eps_b, tmp_path)
            n_second = len(calls)
    # Second proxy must issue fresh judge calls (not reuse first proxy's cache).
    assert n_first > 0
    assert n_second > n_first
    keys = cache.load_pickle(tmp_path, "turing_judgments.pkl")
    assert len(keys) >= n_second
    assert any("proxy-AAA" not in k and "proxy-BBB" not in k for k in keys)  # keys are hashes
    # Distinct pair hashes appear in keys.
    pair_hashes = {k.split("|")[4] for k in keys}
    assert len(pair_hashes) >= 2


def test_turing_score_formula_and_orderings(tmp_path: Path):
    from cue_training.evaluation.metrics import cache

    baseline = {
        f"e{i}": BaselineEpisode(
            episode_id=f"e{i}",
            human=[{"role": "user", "content": f"human text {i} " * 20}],
            base=[{"role": "user", "content": f"base text {i} " * 20}],
            metadata={"domain": "airline"},
            raw={},
        )
        for i in range(20)
    }
    api_metrics.fit_turing_state(baseline, tmp_path, n_eval=6, n_fewshot=2, seed=0)
    eval_ids = cache.load_pickle(tmp_path, "turing_eval_ids.pkl")
    assert "airline" in eval_ids
    few = cache.load_pickle(tmp_path, "turing_few_shot.pkl")
    for text, label in few["airline"]:
        assert label in (0, 1)
        assert text

    eps = [
        _ep(eid, arm="paired", domain="airline")
        for eid in eval_ids["airline"]
    ]

    def _fake_batch(reqs, *, concurrency=None):
        # Always pick the human side: reply A when human is first, B when proxy is first.
        out = []
        for model, prompt, _mt in reqs:
            a_block = prompt.split("A:\n", 1)[-1].split("\n\nB:", 1)[0]
            out.append("A" if "human-" in a_block else "B")
        return out

    with patch.object(api_metrics, "completion_batch", side_effect=_fake_batch):
        with patch.object(api_metrics, "_available_judges", return_value=api_metrics.TURING_JUDGES):
            agg = api_metrics.turing_sonnet_qwen(eps, tmp_path)
    # Perfect human preference → corpus P̄=1 → |0.5-1|=0.5
    assert agg.sample_size == 1
    assert abs(agg.mean - 0.5) < 1e-6
    assert agg.extras.get("mean_p_human") == pytest.approx(1.0)
    assert agg.extras.get("aggregation") == "corpus_|0.5-mean_P|"


def test_turing_corpus_aggregation_cancels_balanced_preferences(tmp_path: Path):
    """Half the episodes favor human, half favor proxy → corpus |0.5-P̄|=0 (not 0.5)."""

    from cue_training.evaluation.metrics import cache

    baseline = {
        f"e{i}": BaselineEpisode(
            episode_id=f"e{i}",
            human=[{"role": "user", "content": f"human text {i} " * 20}],
            base=[{"role": "user", "content": f"base text {i} " * 20}],
            metadata={"domain": "airline"},
            raw={},
        )
        for i in range(20)
    }
    api_metrics.fit_turing_state(baseline, tmp_path, n_eval=6, n_fewshot=2, seed=0)
    eval_ids = cache.load_pickle(tmp_path, "turing_eval_ids.pkl")["airline"]
    eps = []
    for i, eid in enumerate(eval_ids):
        idx = int(eid[1:]) if eid[1:].isdigit() else i
        tag = "EVEN" if i % 2 == 0 else "ODD"
        eps.append(
            _ep(
                eid,
                arm="paired",
                domain="airline",
                human=[f"human text {idx} " * 20],
                proxy=[f"proxy-{tag} {i} " * 20],
            )
        )

    def _fake_batch(reqs, *, concurrency=None):
        out = []
        for _model, prompt, _mt in reqs:
            a_block = prompt.split("A:\n", 1)[-1].split("\n\nB:", 1)[0]
            a_is_human = "human text" in a_block
            # EVEN proxy → always pick proxy; ODD proxy → always pick human.
            prefer_human = "proxy-ODD" in prompt
            if prefer_human:
                out.append("A" if a_is_human else "B")
            else:
                out.append("B" if a_is_human else "A")
        return out

    with patch.object(api_metrics, "completion_batch", side_effect=_fake_batch):
        with patch.object(api_metrics, "_available_judges", return_value=[{"name": "j", "model": "m"}]):
            agg = api_metrics.turing_sonnet_qwen(eps, tmp_path)

    # Balanced preferences → corpus P̄ ≈ 0.5 → score ≈ 0 (per-episode agg would be ~0.5).
    assert agg.mean == pytest.approx(0.0, abs=1e-6)
    assert agg.extras.get("mean_p_human") == pytest.approx(0.5, abs=1e-6)


def test_turing_eval_ids_use_real_episode_id_under_aggregate_fit(tmp_path: Path):
    """Aggregate fitting keys baselines as ``<id>__<sim>``; scoring joins on the plain id."""

    from cue_training.evaluation.metrics import cache

    baseline = {
        f"e{i}__gpt": BaselineEpisode(
            episode_id=f"e{i}__gpt",
            human=[{"role": "user", "content": f"human text {i} " * 20}],
            base=[{"role": "user", "content": f"base text {i} " * 20}],
            metadata={"domain": "airline", "aggregate_source_episode_id": f"e{i}"},
            raw={},
        )
        for i in range(20)
    }
    api_metrics.fit_turing_state(baseline, tmp_path, n_eval=6, n_fewshot=2, seed=0)
    eval_ids = cache.load_pickle(tmp_path, "turing_eval_ids.pkl")
    assert all("__" not in eid for eid in eval_ids["airline"])

    # Stale state written with suffixed ids must still resolve against plain candidates.
    eps = [_ep(f"e{i}", arm="paired", domain="airline") for i in range(20)]
    resolved = api_metrics._resolve_eval_episodes(["e3__gpt", "e7"], eps)
    assert [eid for eid, _ in resolved] == ["e3__gpt", "e7"]
    assert [ep.episode_id for _, ep in resolved] == ["e3", "e7"]


def test_paired_audit_five_dims(tmp_path: Path):
    baseline = {
        f"e{i}": BaselineEpisode(
            episode_id=f"e{i}",
            human=[{"role": "user", "content": f"human {i}"}],
            base=[{"role": "user", "content": f"base {i}"}],
            metadata={"domain": "airline"},
            raw={},
        )
        for i in range(12)
    }
    api_metrics.fit_turing_state(baseline, tmp_path, n_eval=4, n_fewshot=2, seed=1)
    from cue_training.evaluation.metrics import cache

    ids = cache.load_pickle(tmp_path, "turing_eval_ids.pkl")["airline"]
    eps = [_ep(eid) for eid in ids]

    def _fake_batch(reqs, *, concurrency=None):
        body = "\n".join(f"{k}: MATCH" for k, _ in api_metrics.PAIRED_AUDIT_DIMENSIONS)
        return [body for _ in reqs]

    with patch.object(api_metrics, "completion_batch", side_effect=_fake_batch):
        with patch.object(api_metrics, "_available_judges", return_value=api_metrics.TURING_JUDGES):
            agg = api_metrics.paired_audit_fidelity(eps, tmp_path)
    assert agg.mean == pytest.approx(1.0)
    assert all(v == pytest.approx(1.0) for v in (agg.extras.get("per_dimension") or {}).values())


def test_paired_audit_blank_replies_count_as_failures(tmp_path: Path):
    """A blank/garbled judge reply must not score as all-NO_MATCH (fidelity 0)."""

    baseline = {
        f"e{i}": BaselineEpisode(
            episode_id=f"e{i}",
            human=[{"role": "user", "content": f"human {i}"}],
            base=[{"role": "user", "content": f"base {i}"}],
            metadata={"domain": "airline"},
            raw={},
        )
        for i in range(12)
    }
    api_metrics.fit_turing_state(baseline, tmp_path, n_eval=4, n_fewshot=2, seed=1)
    from cue_training.evaluation.metrics import cache

    ids = cache.load_pickle(tmp_path, "turing_eval_ids.pkl")["airline"]
    eps = [_ep(eid) for eid in ids]

    body = "\n".join(f"{k}: MATCH" for k, _ in api_metrics.PAIRED_AUDIT_DIMENSIONS)

    def _half_blank(reqs, *, concurrency=None):
        return [("" if i % 2 else body) for i in range(len(reqs))]

    with patch.object(api_metrics, "completion_batch", side_effect=_half_blank):
        with patch.object(api_metrics, "_available_judges", return_value=api_metrics.TURING_JUDGES):
            agg = api_metrics.paired_audit_fidelity(eps, tmp_path)

    # Only the parseable half votes, so fidelity stays 1.0 and blanks surface as failures.
    assert agg.mean == pytest.approx(1.0)
    assert agg.extras["n_failed_calls"] > 0
    # Blanks are never persisted, so they cannot poison later reruns.
    assert all(str(v).strip() for v in cache.load_pickle(tmp_path, "paired_audit_judgments.pkl").values())


def test_turing_blank_replies_are_not_ties(tmp_path: Path):
    """Blank replies must be failures, not TIE votes that drag the score toward 0."""

    baseline = {
        f"e{i}": BaselineEpisode(
            episode_id=f"e{i}",
            human=[{"role": "user", "content": f"human text {i} " * 20}],
            base=[{"role": "user", "content": f"base text {i} " * 20}],
            metadata={"domain": "airline"},
            raw={},
        )
        for i in range(20)
    }
    api_metrics.fit_turing_state(baseline, tmp_path, n_eval=6, n_fewshot=2, seed=0)
    from cue_training.evaluation.metrics import cache

    eval_ids = cache.load_pickle(tmp_path, "turing_eval_ids.pkl")["airline"]
    eps = [_ep(eid, arm="paired", domain="airline") for eid in eval_ids]

    def _blank(reqs, *, concurrency=None):
        return ["" for _ in reqs]

    with patch.object(api_metrics, "completion_batch", side_effect=_blank):
        with patch.object(api_metrics, "_available_judges", return_value=api_metrics.TURING_JUDGES):
            agg = api_metrics.turing_sonnet_qwen(eps, tmp_path)

    assert agg.extras["n_scored"] == 0
    assert agg.extras["n_failed_calls"] > 0
    assert agg.extras.get("skipped")


def test_sim2real_chamfer_uses_ppol_fingerprints_without_pca():
    """Original PPol coverage: 19-D fingerprints, raw L2 Chamfer (no PCA)."""

    from cue_training.baselines.ppol.features import N_FEATURES, fingerprint

    eps = [
        _ep(f"a{i}", arm="as_is", domain="airline", human=[f"hello there {i} please"], proxy=[f"hi {i}"])
        for i in range(12)
    ] + [
        _ep(f"r{i}", arm="as_is", domain="retail", human=[f"thanks retail {i}"], proxy=[f"ok shop {i}"])
        for i in range(12)
    ]
    out = C.coverage_sim2real_behavioral(eps)
    assert "as_is/airline" in out and "as_is/retail" in out
    for key, agg in out.items():
        assert agg.extras.get("embedding") == "ppol_fingerprint"
        assert agg.extras.get("pca") is False
        assert agg.extras.get("pca_dim") is None
        assert agg.extras.get("n_human", 0) >= 2
        assert 0.0 <= float(agg.mean) <= 1.0
    # Encode path matches fingerprint width.
    x = np.asarray([fingerprint(ep.human) for ep in eps[:3]], dtype=np.float32)
    assert x.shape[1] == N_FEATURES


def test_styledistance_behavioral_splits_by_domain(monkeypatch):
    eps = [
        _ep(f"a{i}", arm="as_is", domain="airline") for i in range(25)
    ] + [
        _ep(f"r{i}", arm="as_is", domain="retail") for i in range(25)
    ]

    def _fake_encode(episodes, *, model_name, side):
        return np.random.default_rng(0).normal(size=(len(episodes), 8)).astype(np.float32)

    monkeypatch.setattr(C, "_encode_styledistance_concat", _fake_encode)
    monkeypatch.setattr(C, "get_styledistance_encoder", lambda *_a, **_k: object())

    out = C.coverage_styledistance_behavioral(eps, pca_dim=4)
    # Airline and retail are scored independently rather than pooled.
    assert "as_is/airline" in out
    assert "as_is/retail" in out
    assert out["as_is/airline"].extras.get("domain") == "airline"
    assert out["as_is/retail"].extras.get("domain") == "retail"
    assert out["as_is/airline"].extras.get("pca") is True
    assert out["as_is/airline"].extras.get("pca_dim") == 4
    assert out["as_is/airline"].extras.get("pca_fit") == "human_reference_only"


def test_realusersim_paired_noex_strips_examples():
    from cue_training.baselines.realusersim.format_persona import format_persona_block, strip_profile_examples
    from cue_training.evaluation.rollouts.recipes import RECIPES, expand_methods

    assert expand_methods("baseline", ["realusersim-paired-noex"]) == ["realusersim_paired_noex"]
    recipe = RECIPES[("baseline", "realusersim_paired_noex")]
    assert recipe.default_arms == ("paired",)

    profile = {
        "command_examples": [
            {"command": "book flight", "examples": ["please book my flight to NYC"]},
        ],
        "include_examples": True,
    }
    stripped = strip_profile_examples(profile)
    assert stripped["command_examples"][0]["examples"] == []
    block = format_persona_block(stripped, include_examples=False)
    assert "Examples:" not in block
    assert "book flight" in block


def test_authenticity_ava_alias_dispatches_to_wegmann(monkeypatch):
    calls = {}

    def _fake_mimicry(episodes, *, backend, kind, luar_model=None):
        calls["backend"] = backend
        calls["kind"] = kind
        from cue_training.evaluation.metrics.stats import aggregate

        return aggregate("mimicry/wegmann_ava", [100.0], {"ok": True})

    monkeypatch.setattr(authenticity_metrics, "mimicry_similarity", _fake_mimicry)
    agg = authenticity_metrics.run_authenticity_metric("authenticity/ava", [_ep("x")])
    assert calls == {"backend": "wegmann", "kind": "ava"}
    assert agg.metric_name == "authenticity/ava"
    assert agg.extras.get("alias_of") == "mimicry/wegmann_ava"
