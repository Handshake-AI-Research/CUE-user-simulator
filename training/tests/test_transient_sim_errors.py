"""Provider blips must be retried / tolerated; config errors must still fail closed."""

from __future__ import annotations

import io
import json
import urllib.request
from http.client import RemoteDisconnected

import pytest

from cue_training.evaluation.common import user_sims
from cue_training.evaluation.common.user_sims import (
    BaselineSimError,
    SidecarError,
    is_transient_provider_error,
    post_sidecar_json,
)


class APIError(Exception):
    """Stand-in for litellm's generic APIError (class name carries no signal)."""


class APIConnectionError(Exception):
    pass


class BadRequestError(Exception):
    pass


def test_openrouter_disconnect_is_transient():
    exc = APIError(
        "litellm.APIError: APIError: OpenrouterException - "
        "Server disconnected without sending a response."
    )
    assert is_transient_provider_error(exc)


def test_wrapped_sidecar_error_is_transient():
    exc = SidecarError(
        "CUE sidecar returned error (episode=gpt-4-turbo_f222c5ef_email): "
        "SidecarError('CUE decoder sim call failed: litellm.APIError: "
        "OpenrouterException - Server disconnected without sending a response.')"
    )
    assert is_transient_provider_error(exc)


def test_error_class_names_still_match():
    assert is_transient_provider_error(APIConnectionError("boom"))


def test_config_errors_are_not_transient():
    assert not is_transient_provider_error(BadRequestError("model not found: gpt-9"))
    assert not is_transient_provider_error(ValueError("missing OPENROUTER_API_KEY"))


def test_episode_id_digits_do_not_false_positive():
    # Episode ids are interpolated into these messages; a bare "502" must not match.
    exc = SidecarError("CUE sidecar returned error (episode=abc502def): decoder weights missing")
    assert not is_transient_provider_error(exc)


class _Resp(io.BytesIO):
    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()
        return False


@pytest.fixture
def no_sleep(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(user_sims.time, "sleep", lambda _s: None)


def test_predecode_post_retries_remote_disconnect(monkeypatch: pytest.MonkeyPatch, no_sleep):
    calls = {"n": 0}

    def fake_urlopen(req, timeout=None):
        calls["n"] += 1
        if calls["n"] < 3:
            raise RemoteDisconnected("Remote end closed connection without response")
        return _Resp(json.dumps({"manuals": {"k": "manual"}}).encode("utf-8"))

    monkeypatch.setattr(urllib.request, "urlopen", fake_urlopen)
    body = post_sidecar_json("http://127.0.0.1:18700/decode_manuals", {"items": []}, timeout=10)
    assert body["manuals"] == {"k": "manual"}
    assert calls["n"] == 3


def test_predecode_post_gives_up_on_dead_sidecar(monkeypatch: pytest.MonkeyPatch, no_sleep):
    def fake_urlopen(req, timeout=None):
        raise RemoteDisconnected("Remote end closed connection without response")

    monkeypatch.setattr(urllib.request, "urlopen", fake_urlopen)
    with pytest.raises(SidecarError, match="after 3 attempts"):
        post_sidecar_json("http://127.0.0.1:18700/decode_manuals", {"items": []}, timeout=10)


@pytest.fixture(autouse=True)
def reset_baseline_failures(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(user_sims, "_BASELINE_SIM_FAILURES", 0)


def test_scattered_baseline_failures_are_tolerated():
    limit = user_sims._BASELINE_SIM_MAX_CONSECUTIVE_FAILURES
    for _ in range(limit * 2):
        user_sims._note_baseline_sim_failure("realusersim", RuntimeError("blip"))
        user_sims._note_baseline_sim_success()


def test_systemic_baseline_failures_abort():
    limit = user_sims._BASELINE_SIM_MAX_CONSECUTIVE_FAILURES
    exc = RuntimeError("get_supported_openai_params() got an unexpected keyword argument")
    for _ in range(limit - 1):
        user_sims._note_baseline_sim_failure("realusersim", exc)
    with pytest.raises(BaselineSimError, match="consecutive user-sim failures"):
        user_sims._note_baseline_sim_failure("realusersim", exc)
