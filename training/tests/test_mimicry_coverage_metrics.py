"""Default metrics + StyleDistance mimicry/coverage plumbing (no model downloads)."""

from __future__ import annotations

import numpy as np
import pytest

from cue_training.evaluation.metrics.data import BaselineEpisode, Episode, _metadata
from cue_training.evaluation.metrics.metrics import coverage as C
from cue_training.evaluation.metrics.metrics import local as local_metrics
from cue_training.evaluation.metrics.run import ALL_METRICS, DEFAULT_METRICS, _parse_metrics


def _ep(
    eid: str,
    *,
    arm: str,
    domain: str = "airline",
    human: list[str] | None = None,
    proxy: list[str] | None = None,
    origin: str | None = None,
) -> Episode:
    human = human or [f"human-{eid}-a", f"human-{eid}-b"]
    proxy = proxy or [f"proxy-{eid}-a", f"proxy-{eid}-b"]
    meta: dict = {"arm": arm, "domain": domain}
    if origin is not None:
        meta["persona_origin_episode_id"] = origin
    elif arm == "paired":
        meta["persona_origin_episode_id"] = eid
    return Episode(
        episode_id=eid,
        human=[{"role": "user", "content": t} for t in human] + [{"role": "assistant", "content": "ok"}],
        base=[{"role": "user", "content": "base"}],
        proxy=[{"role": "user", "content": t} for t in proxy] + [{"role": "assistant", "content": "ok"}],
        metadata=meta,
        base_metadata={"domain": domain},
        arm=arm,
        domain=domain,
    )


def test_default_metrics_are_the_paper_suite():
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
    assert ALL_METRICS == DEFAULT_METRICS
    assert _parse_metrics("default") == DEFAULT_METRICS
    assert _parse_metrics("all") == ALL_METRICS


def test_aliases():
    assert _parse_metrics("coverage") == [
        "coverage/styledistance_behavioral",
        "coverage/sim2real_behavioral",
    ]
    assert _parse_metrics("ava") == ["mimicry/wegmann_ava"]
    assert _parse_metrics("styledistance_behavioral") == ["coverage/styledistance_behavioral"]
    assert _parse_metrics("sim2real_behavioral") == ["coverage/sim2real_behavioral"]
    assert _parse_metrics("sim2real_classifier") == ["classifier/sim2real"]
    assert _parse_metrics("task_success") == ["env/tau2_task_success"]
    assert _parse_metrics("env_success") == ["env/tau2_success_rate"]


def test_persona_origin_inference_in_metadata():
    paired = _metadata({"episode_id": "e1", "arm": "paired", "real_conversation": [{"role": "user", "content": "hi"}]})
    assert paired["persona_origin_episode_id"] == "e1"
    shuffled = _metadata(
        {
            "episode_id": "e1",
            "arm": "sample_shuffled",
            "shuffled_episode_id": "e9",
            "real_conversation": [{"role": "user", "content": "hi"}],
        }
    )
    assert shuffled["persona_origin_episode_id"] == "e9"


def test_tau2_task_success_f1_and_env_rate(monkeypatch):
    monkeypatch.setattr(
        local_metrics.tau_usi,
        "human_rewards",
        lambda _src=None: {"e1": 1.0, "e2": 0.0, "e3": 1.0},
    )
    eps = [
        _ep("e1", arm="paired", domain="airline"),
        _ep("e2", arm="as_is", domain="airline"),
        _ep("e3", arm="sample_shuffled", domain="airline", origin="e3"),
    ]
    eps[0].metadata.update({"is_success": True, "reward": 1.0})
    eps[1].metadata.update({"is_success": False, "reward": 0.0})
    eps[2].metadata.update({"is_success": False, "reward": 0.0})
    out = local_metrics.tau2_task_success(eps)
    assert out.metric_name == "env/tau2_task_success"
    assert out.mean == pytest.approx(2 / 3)  # TP=1, FP=0, FN=1 -> F1=2/3
    assert out.confidence_interval is not None
    assert out.extras["env_success_rate"]["n"] == 3
    assert out.extras["env_success_rate"]["mean"] == pytest.approx(1 / 3)
    assert out.extras["pairwise_success_agreement"] == pytest.approx(2 / 3, rel=1e-3)


def test_styledistance_behavioral_coverage_groups_tasks_and_scores_all_arms(monkeypatch):
    class FakeEnc:
        def encode(self, texts):
            rows = []
            for text in texts:
                if "human-e0" in text or "proxy-left" in text:
                    value = 1.0
                elif "human-e1" in text or "proxy-right" in text:
                    value = 5.0
                else:
                    value = 3.0
                rows.append([value])
            return np.asarray(rows, dtype=np.float32)

    monkeypatch.setattr(C, "get_styledistance_encoder", lambda *a, **k: FakeEnc())
    paired = [
        _ep("e0", arm="paired", proxy=["proxy-left"]),
        _ep("e1", arm="paired", proxy=["proxy-right"]),
    ]
    for ep in paired:
        ep.metadata["task_id"] = "shared-task"
    shuffled = [_ep("e2", arm="sample_shuffled", proxy=["proxy-mid"])]

    out = C.coverage_styledistance_behavioral(paired + shuffled, pca_dim=1)
    assert out["paired"].mean == pytest.approx(0.875)
    assert out["paired"].sample_size == 1
    assert out["paired"].extras["aggregation"] == "domain_level_chamfer"
    assert out["paired"].extras["n_proxy"] == 2
    assert out["paired"].extras["pca_fit"] == "human_reference_only"
    assert out["sample_shuffled"].mean == pytest.approx(0.75)
    assert out["sample_shuffled"].extras["reference"] == "rollout_humans"