"""Shared CUE-side HTTP sidecar: serve user-turn generation from a GPU node.

Runs in the CUE venv (Python 3.13). Lets the harness driver processes stay CPU-only:
they call this service instead of loading the decoder (or local baseline
models) in-process. Generation is serialized with a global lock (the decoder shares one
process-global GPU model via the model hub). Stdlib-only (http.server) -- no extra deps.

Endpoints:
- ``POST /next_turn`` (tau2 / simulatorarena): the request carries the full per-episode
  conditioning (``variant``, ``arm``, ``episode_id``, ``task``, ``domain``, ``cue_embedding``,
  ``persona``) plus the ``history``; returns ``{text, done}``.
- ``POST /reset``: drop the cached sim for a (variant, arm, episode) key.

Model paths (adapters, decoder dir, baseline dirs, base-local path) are owned by the
sidecar CLI args; requests carry only conditioning.
"""

from __future__ import annotations

import argparse
import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any

from cue_training.utils.config import storage_root
from cue_training.runlog.log import log, warn

_GEN_LOCK = threading.Lock()   # serializes single-stream HF/GPU generation
_BUILD_LOCK = threading.Lock()  # serializes sim construction (model loads)
_PREDECODE_LOCK = threading.Lock()  # serializes decoder load/cache init + batch generation
_SIMS: dict[tuple, Any] = {}
_ARGS: argparse.Namespace | None = None


def _extra_for(variant: str) -> dict[str, Any]:
    a = _ARGS
    common = {"device": a.device, "dtype": a.dtype}
    if variant == "decoder":
        extra = {**common, "decoder_dir": a.decoder_dir, "semantic_model": a.semantic_model,
                 "sim_model": a.sim_model, "sim_api_base": a.sim_api_base,
                 "sim_api_key_env": a.sim_api_key_env,
                 "decode_temperature": getattr(a, "decode_temperature", 0.0),
                 "decode_top_p": getattr(a, "decode_top_p", 1.0),
                 "decode_num_candidates": getattr(a, "decode_num_candidates", 1),
                 "decode_mode": getattr(a, "decode", "sample"),
                 "decode_slot_dedup_jaccard": getattr(a, "decode_slot_dedup_jaccard", 0.5),
                 "decode_noop_retries": getattr(a, "decode_noop_retries", 0),
                 "decode_noop_temperature": getattr(a, "decode_noop_temperature", 0.7),
                 "example_retrieval_enabled": getattr(a, "example_retrieval_enabled", False),
                 "example_retrieval_k_sessions": getattr(a, "example_retrieval_k_sessions", 8),
                 "example_retrieval_n_general": getattr(a, "example_retrieval_n_general", 2),
                 "example_retrieval_n_specific": getattr(a, "example_retrieval_n_specific", 2)}
        if getattr(a, "decoder_vllm_base_url", None):  # offload the command-block decode to vLLM
            extra["decoder_vllm_base_url"] = a.decoder_vllm_base_url
            extra["decoder_vllm_model"] = a.decoder_vllm_model
            extra["decoder_vllm_api_key_env"] = a.vllm_api_key_env
        return extra
    if variant.startswith("baseline:"):
        extra = {**common, "name": variant.split(":", 1)[1], "output_dir": a.baseline_output_dir,
                 "artifacts_dir": a.baseline_artifacts_dir, "sim_model": a.sim_model,
                 "sim_api_base": a.sim_api_base, "sim_api_key_env": a.sim_api_key_env}
        if getattr(a, "usp_vllm_base_url", None):  # serve published USP via vLLM (batched)
            extra["usp_vllm_base_url"] = a.usp_vllm_base_url
            extra["usp_vllm_model"] = a.usp_vllm_model
            extra["usp_vllm_api_key_env"] = a.vllm_api_key_env
        if getattr(a, "userlm_vllm_base_url", None):  # serve UserLM-8b via vLLM (batched)
            extra["userlm_vllm_base_url"] = a.userlm_vllm_base_url
            extra["userlm_vllm_model"] = a.userlm_vllm_model
            extra["userlm_vllm_api_key_env"] = a.vllm_api_key_env
        return extra
    if variant == "base_local":
        extra = {**common, "hf_path": a.base_local_path}
        if getattr(a, "vllm_base_url", None):  # serve base_local via a local vLLM endpoint
            extra["vllm_base_url"] = a.vllm_base_url
            extra["vllm_api_key_env"] = a.vllm_api_key_env
        return extra
    return {"sim_model": a.sim_model}


def _sim_from_ctx(variant: str, arm: str, episode_id: str, *, task: str, domain: str | None,
                  cue_embedding: Any, persona: Any, system_prompt: str | None = None,
                  request_extra: dict[str, Any] | None = None):
    request_extra = request_extra or {}
    key_extra = {
        k: request_extra.get(k)
        for k in (
            # sim_model/sim_api_base distinguish simulators sharing ONE sidecar (e.g. cue-general
            # over llama+gpt): the decoded manual is shared, but each builds its own generator sim.
            "sim_model", "sim_api_base", "command_block",
        )
        if k in request_extra
    }
    key = (variant, arm, episode_id, json.dumps(key_extra, sort_keys=True))
    if key in _SIMS:
        return _SIMS[key]
    from cue_training.evaluation.common.conditioning import TurnContext
    from cue_training.evaluation.common.user_sims import build_user_sim

    with _BUILD_LOCK:
        if key in _SIMS:
            return _SIMS[key]
        extra = _extra_for(variant)
        # Only let the request override with real values -- a None (e.g. decoder_dir the client
        # didn't set) must NOT clobber the sidecar's launch config, or the decoder loads Path(None).
        extra.update({k: v for k, v in request_extra.items() if v is not None})
        # Shared cue-general sidecar is often started by the llama job with sim_api_base=vLLM.
        # GPT/Gemini clients then send sim_model=gpt-5.4-mini (etc.) with sim_api_base=None meaning
        # "provider default". Skipping None left the vLLM URL sticky, so litellm called the local
        # Llama server and 404'd ("model does not exist" / Gemini {"detail":"Not Found"}).
        if "sim_api_base" in request_extra:
            extra["sim_api_base"] = request_extra.get("sim_api_base")
        # The client forwards its whole extra (incl. sidecar_url); drop it here so build_user_sim
        # constructs the REAL local GPU sim instead of another RemoteUserSim that POSTs back to us
        # (self-call deadlock: the recursive /next_turn blocks on the response while holding _GEN_LOCK).
        extra.pop("sidecar_url", None)
        ctx = TurnContext(
            variant=variant, arm=arm, task=task or "", domain=domain, episode_id=episode_id,
            cue_embedding=cue_embedding, persona=persona,
            # The CUE decoder's user simulator builds on the harness's standard prompt (which carries
            # the task instructions): the decoder injects its command block into it (grounds the
            # task so the sim doesn't drift off-task). base/baselines keep their prior prompt, so
            # gate to the decoder variant only.
            system_prompt=(system_prompt if variant == "decoder" else None),
            extra=extra,
        )
        sim = build_user_sim(ctx)
        _SIMS[key] = sim
    return sim


def _next_turn(payload: dict[str, Any]) -> dict[str, Any]:
    variant = payload.get("variant", "decoder")
    arm = payload.get("arm", "paired")
    episode_id = str(payload.get("episode_id") or payload.get("task_name") or "")
    history = payload.get("history") or []
    # Reject obviously-empty/invalid requests (e.g. a stray probe) before building or invoking
    # a GPU sim, so we don't load the model or raise a cryptic tensor error.
    request_extra = payload.get("extra") or {}
    if (
        variant == "decoder"
        and payload.get("cue_embedding") is None
        and not request_extra.get("command_block")
    ):
        return {"text": "", "done": True, "error": "decoder request missing cue_embedding"}
    try:
        sim = _sim_from_ctx(
            variant, arm, episode_id, task=payload.get("task", ""), domain=payload.get("domain"),
            cue_embedding=payload.get("cue_embedding"), persona=payload.get("persona"),
            system_prompt=payload.get("system_prompt"), request_extra=request_extra,
        )
        # The decoder sim self-serializes its GPU decode via _DECODER_GPU_LOCK and then does the
        # heavy steerable-LLM turn over the network; wrapping it in the coarse _GEN_LOCK too would
        # serialize that network call across all concurrent episodes (64 clients queue behind one
        # lock until they hit their timeout -> "client disconnected before response"). Skip the
        # coarse lock for the decoder (and vLLM-batched sims); keep it for other in-process HF sims
        # that generate on the GPU without their own lock.
        if getattr(sim, "batched", False) or variant == "decoder":
            text, done = sim.next_turn(history)
        else:
            with _GEN_LOCK:  # single-stream HF/GPU generation
                text, done = sim.next_turn(history)
    except Exception as exc:  # noqa: BLE001
        warn("sidecar", f"next_turn failed ({variant}/{arm}/{episode_id}): {exc}")
        return {"text": "", "done": True, "error": repr(exc)}
    # Per-turn provenance (e.g. the decoder's injected command block) for rollout metadata.
    return {"text": (text or "").strip(), "done": bool(done), "command": getattr(sim, "last_command", None)}


def _decode_manuals(payload: dict[str, Any]) -> dict[str, Any]:
    """Batch-decode + cache persona manuals for a whole dataset up front, so subsequent
    /next_turn calls skip the GPU decode. Items contain either a live trajectory or a
    backward-compatible cue embedding."""

    items = [
        (
            str(it.get("key")),
            it.get("trajectory")
            if it.get("trajectory") is not None
            else it.get("cue_embedding"),
        )
        for it in (payload.get("items") or [])
        if it.get("trajectory") is not None or it.get("cue_embedding") is not None
    ]
    if not items:
        return {"decoded": 0}
    decoder_dir = str(payload.get("decoder_dir") or getattr(_ARGS, "decoder_dir", "") or "")
    try:
        from cue_training.evaluation.common.user_sims import decode_manuals_batch

        # Parallel jobs sharing this sidecar (e.g. llama + GPT) reach this endpoint together.
        # Serialize the whole operation, including model_hub.get_decoder inside the helper:
        # model-hub cache initialization is not thread-safe, and racing two model builds can leave
        # one request decoding with a partially initialized/corrupted model.
        with _PREDECODE_LOCK:
            out = decode_manuals_batch(
                decoder_dir=decoder_dir, items=items,
                device=str(payload.get("device") or "cuda"),
                dtype=str(payload.get("dtype") or "bfloat16"),
                max_new_tokens=int(payload.get("max_new_tokens") or 1024),
                temperature=float(payload.get("temperature") or 0.0),
                top_p=float(payload.get("top_p") or 1.0),
                batch_size=int(payload.get("batch_size") or 16),
                num_candidates=int(payload.get("num_candidates") or 1),
                seed=(int(payload["seed"]) if payload.get("seed") is not None else None),
                decode_mode=str(payload.get("decode_mode") or "sample"),
                decode_slot_dedup_jaccard=float(
                    payload.get("decode_slot_dedup_jaccard") or 0.5
                ),
                decode_noop_retries=int(payload.get("decode_noop_retries") or 0),
                decode_noop_temperature=float(
                    payload.get("decode_noop_temperature") or 0.7
                ),
                example_retrieval_enabled=bool(payload.get("example_retrieval_enabled")),
                example_retrieval_k_sessions=int(payload.get("example_retrieval_k_sessions") or 8),
                example_retrieval_n_general=int(payload.get("example_retrieval_n_general") or 2),
                example_retrieval_n_specific=int(payload.get("example_retrieval_n_specific") or 2),
                session_preprocess=str(payload.get("session_preprocess") or "full"),
            )
    except Exception as exc:  # noqa: BLE001
        warn("sidecar", f"decode_manuals failed: {exc}")
        return {"error": repr(exc), "decoded": 0}
    empty = [key for key, manual in out.items() if not manual.strip()]
    return {"decoded": len(out) - len(empty), "manuals": out, "empty": empty}


class _Handler(BaseHTTPRequestHandler):
    def _send(self, code: int, body: dict[str, Any]) -> None:
        data = json.dumps(body).encode("utf-8")
        try:
            self.send_response(code)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)
        except (BrokenPipeError, ConnectionResetError):
            # The client already gave up (e.g. hit its sidecar_timeout) and closed the socket
            # before we finished the (slow) turn. Nothing to send; drop it quietly instead of
            # dumping a traceback per abandoned request.
            warn("sidecar", "client disconnected before response (timed out?)")

    def log_message(self, *_args) -> None:
        return

    def do_GET(self) -> None:  # noqa: N802
        # Lightweight liveness probe that never touches the GPU/model.
        if self.path.rstrip("/") == "/health":
            self._send(200, {"ok": True})
        else:
            self._send(404, {"error": "unknown path"})

    def do_POST(self) -> None:  # noqa: N802
        length = int(self.headers.get("Content-Length", 0))
        try:
            payload = json.loads(self.rfile.read(length) or b"{}")
        except json.JSONDecodeError:
            self._send(400, {"error": "invalid json"})
            return
        path = self.path.rstrip("/")
        if path == "/next_turn":
            self._send(200, _next_turn(payload))
        elif path == "/decode_manuals":
            self._send(200, _decode_manuals(payload))
        elif path == "/reset":
            _SIMS.pop((payload.get("variant"), payload.get("arm"),
                       str(payload.get("episode_id") or payload.get("task_name") or "")), None)
            self._send(200, {"ok": True})
        else:
            self._send(404, {"error": "unknown path"})


def serve(args: argparse.Namespace) -> None:
    global _ARGS
    _ARGS = args
    server = ThreadingHTTPServer((args.host, args.port), _Handler)
    log("sidecar", f"serving on http://{args.host}:{args.port}")
    server.serve_forever()


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="CUE sidecar for harness user-turn generation.")
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=8713)
    p.add_argument("--sim_model", default="gpt-5.4-mini")
    p.add_argument("--sim_api_base", default=None,
                   help="OpenAI-compatible base URL for the decoder simulator (e.g. a vLLM "
                        "endpoint serving an open model). Default: the provider API.")
    p.add_argument("--sim_api_key_env", default="OPENAI_API_KEY",
                   help="Env var holding the API key used with --sim_api_base.")
    p.add_argument("--semantic_model", default="intfloat/e5-base-v2")
    p.add_argument("--decoder_dir", default=None)
    p.add_argument("--decode_temperature", type=float, default=0.0,
                   help="Command-block decode temperature for decoder (0.0=greedy; >0 samples).")
    p.add_argument("--decode_top_p", type=float, default=1.0,
                   help="Nucleus sampling probability for decoder generation.")
    p.add_argument("--decode_num_candidates", type=int, default=1,
                   help="Manuals to sample per persona; >1 reranks for naturalness (samples even "
                   "when decode_temperature=0).")
    p.add_argument("--decode", choices=("greedy", "sample", "diverse_slots"),
                   default="sample",
                   help="Manual selection: greedy/sample, or diverse_slots "
                        "(per-slot threshold dedup; decode_num_candidates = samples per slot).")
    p.add_argument("--decode_slot_dedup_jaccard", type=float, default=0.5,
                   help="For --decode diverse_slots: reject candidates with token-Jaccard ≥ this "
                        "vs an already-kept command in the same head.")
    p.add_argument("--decode_noop_retries", type=int, default=0,
                   help="Resample slots that greedily decode to <NO_COMMAND> this many times "
                        "instead of leaving the slot empty.")
    p.add_argument("--decode_noop_temperature", type=float, default=0.7,
                   help="Sampling temperature for --decode_noop_retries.")
    p.add_argument(
        "--example_retrieval_enabled",
        action="store_true",
        help="Inject nearest-neighbor style examples from the joint training example_pool.",
    )
    p.add_argument("--example_retrieval_k_sessions", type=int, default=8)
    p.add_argument("--example_retrieval_n_general", type=int, default=2)
    p.add_argument("--example_retrieval_n_specific", type=int, default=2)
    p.add_argument("--baseline_output_dir", default=str(storage_root() / "baselines"))
    p.add_argument("--baseline_artifacts_dir", default=None)
    p.add_argument("--base_local_path", default="meta-llama/Llama-3.1-8B-Instruct")
    p.add_argument("--vllm_base_url", default=None,
                   help="If set, base_local is served by this vLLM OpenAI endpoint (batched, "
                        "concurrent) instead of an in-process HF model, e.g. http://127.0.0.1:8000/v1")
    p.add_argument("--decoder_vllm_base_url", default=None,
                   help="If set, decoder offloads the command-block DECODE to this vLLM "
                        "--enable-prompt-embeds endpoint serving the merged decoder (see "
                        "cue.decoder.merge), so concurrent episodes' decodes batch there.")
    p.add_argument("--decoder_vllm_model", default="decoder-merged",
                   help="Model name registered with the vLLM server for the merged decoder.")
    p.add_argument("--usp_vllm_base_url", default=None,
                   help="If set, baseline:usp is served by this vLLM OpenAI endpoint (batched) "
                        "for the published HF model wangkevin02/USP, e.g. "
                        "http://127.0.0.1:8000/v1")
    p.add_argument("--usp_vllm_model", default="wangkevin02/USP",
                   help="Model name registered with the vLLM server for USP (default: wangkevin02/USP).")
    p.add_argument("--userlm_vllm_base_url", default=None,
                   help="If set, baseline:userlm is served by this vLLM OpenAI endpoint (batched) "
                        "serving microsoft/UserLM-8b, e.g. http://127.0.0.1:8000/v1. The decoding "
                        "guardrails are approximated via vLLM sampling params.")
    p.add_argument("--userlm_vllm_model", default="userlm",
                   help="Model name registered with the vLLM server for UserLM-8b.")
    p.add_argument("--vllm_api_key_env", default="HOSTED_VLLM_API_KEY")
    p.add_argument("--device", default="cuda")
    p.add_argument("--dtype", default="bfloat16")
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    serve(parse_args(argv))


if __name__ == "__main__":
    main()
