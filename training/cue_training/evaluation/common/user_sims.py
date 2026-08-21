"""Harness-agnostic "produce the next user turn" interface + implementations.

Each `UserSim` takes a `TurnContext` (task goal + arm + conditioning) and, given the
conversation so far as ``[{role, content}]`` (user = our sim, assistant = agent),
returns ``(text, done)`` for the next user turn. Harnesses adapt this: tau2 wraps it in
a ``HalfDuplexUser``; simulatorarena calls it inside its conversation loop.

Variants:
- ``decoder``: CUE decoder (local GPU), conditioned on the CUE embedding
  (+ last-agent-turn embedding).
- ``baseline:<name>``: a baselines simulator; for realusersim/usp an injected
  persona (arms paired/sample_shuffled) overrides native pool sampling.
- ``base_api`` / ``base_local``: a plain user LM (no conditioning) -- the floor.
"""

from __future__ import annotations

import hashlib
import json
import threading
import time
from typing import Any

from cue_training.evaluation.common import model_hub
from cue_training.evaluation.common.conditioning import TurnContext
from cue_training.runlog.log import log, warn

Turn = dict[str, str]

# Local Llama vLLM rollouts use max_model_len=16384; decoder manuals + long writing
# transcripts can otherwise 400 with ContextWindowExceededError.
_DEFAULT_VLLM_MAX_INPUT_TOKENS = 16384


def _fit_sim_messages(
    messages: list[dict[str, str]],
    *,
    extra: dict[str, Any],
    max_tokens: int,
    model: str | None = None,
) -> list[dict[str, str]]:
    """Truncate chat messages so prompt + max_tokens fits the sim context window."""

    max_input = extra.get("sim_max_input_tokens")
    if max_input is None:
        model_s = str(model or extra.get("sim_model") or "")
        if extra.get("sim_api_base") or model_s.startswith("hosted_vllm/") or "hosted_vllm" in model_s:
            max_input = _DEFAULT_VLLM_MAX_INPUT_TOKENS
    if max_input is None:
        return messages
    from cue_training.preprocessing.llm import truncate_messages

    return truncate_messages(
        messages, max_input_tokens=int(max_input), max_tokens=int(max_tokens)
    )


def example_retrieval_cli_flags(
    *,
    enabled: bool,
    k_sessions: int = 8,
    n_general: int = 2,
    n_specific: int = 2,
) -> list[str]:
    """CLI flags that forward example-pool retrieval into tau2/simarena runners."""

    flags = [
        f"--example_retrieval_k_sessions={int(k_sessions)}",
        f"--example_retrieval_n_general={int(n_general)}",
        f"--example_retrieval_n_specific={int(n_specific)}",
    ]
    if enabled:
        flags.append("--example_retrieval_enabled")
    return flags


def example_retrieval_kwargs(source: Any) -> dict[str, Any]:
    """Pull example-retrieval settings from an argparse Namespace or mapping."""

    get = source.get if isinstance(source, dict) else lambda key, default=None: getattr(source, key, default)
    return {
        "example_retrieval_enabled": bool(get("example_retrieval_enabled", False)),
        "example_retrieval_k_sessions": int(get("example_retrieval_k_sessions") or 8),
        "example_retrieval_n_general": int(get("example_retrieval_n_general") or 2),
        "example_retrieval_n_specific": int(get("example_retrieval_n_specific") or 2),
    }


class SidecarError(RuntimeError):
    """Fatal CUE decoder/sidecar failure. Raised (for the ``decoder`` variant) instead of
    silently returning an empty turn, so a broken sidecar aborts the run rather than emitting
    ``###STOP###`` rollouts that look valid."""


# Overload and tail-latency arrive as these litellm/openai error classes. Anything else (an
# unknown model, a 4xx) is a config error that retrying cannot fix, so it still fails at once.
_TRANSIENT_SIM_ERRORS = ("Timeout", "RateLimit", "APIConnection", "ServiceUnavailable", "InternalServer")

# litellm collapses provider-side disconnects and 5xx into a bare ``APIError``, whose class name
# matches nothing above, so classify those by message. Keep the patterns worded (no bare status
# numbers) -- episode ids are interpolated into these strings and would false-positive on "502".
_TRANSIENT_SIM_MESSAGES = (
    "server disconnected",
    "connection reset",
    "connection aborted",
    "connection error",
    "remote protocol error",
    "peer closed connection",
    "incomplete chunked read",
    "service unavailable",
    "temporarily unavailable",
    "bad gateway",
    "gateway timeout",
    "overloaded",
    "please try again",
)


def _is_transient_sim_error(exc: BaseException) -> bool:
    if any(name in type(exc).__name__ for name in _TRANSIENT_SIM_ERRORS):
        return True
    text = str(exc).lower()
    return any(pattern in text for pattern in _TRANSIENT_SIM_MESSAGES)


def is_transient_provider_error(exc: BaseException) -> bool:
    """True for provider blips (disconnect / 5xx / overload) worth tolerating, not config errors."""

    return _is_transient_sim_error(exc)


# Process-global record of CUE sidecar failures. The tau2 harness swallows user-sim exceptions
# (a raised SidecarError becomes a silent STOP), so the tau2 runner instead checks this after the
# batch and aborts. simarena/prism don't swallow, so they propagate the raised SidecarError.
_SIDECAR_ERRORS: list[str] = []


def sidecar_errors() -> list[str]:
    return list(_SIDECAR_ERRORS)


def reset_sidecar_errors() -> None:
    _SIDECAR_ERRORS.clear()


def _record_sidecar_error(msg: str) -> None:
    _SIDECAR_ERRORS.append(msg)


# A baseline sim that fails returns ("", True), i.e. an episode that *looks* like a complete
# short conversation. Scattered failures are fine, but a broken environment (bad API key, a
# litellm version mismatch) fails every call and would otherwise write a whole run of empty
# rollouts that scores as valid. Abort once failures are clearly systemic.
_BASELINE_SIM_MAX_CONSECUTIVE_FAILURES = 10
_BASELINE_SIM_FAILURES = 0
_BASELINE_SIM_LOCK = threading.Lock()


class BaselineSimError(RuntimeError):
    """Every baseline user-sim call is failing; the environment is broken, not the episode."""


def _note_baseline_sim_success() -> None:
    global _BASELINE_SIM_FAILURES
    with _BASELINE_SIM_LOCK:
        _BASELINE_SIM_FAILURES = 0


def _note_baseline_sim_failure(name: str, exc: Exception) -> None:
    global _BASELINE_SIM_FAILURES
    with _BASELINE_SIM_LOCK:
        _BASELINE_SIM_FAILURES += 1
        count = _BASELINE_SIM_FAILURES
    if count >= _BASELINE_SIM_MAX_CONSECUTIVE_FAILURES:
        raise BaselineSimError(
            f"{name}: {count} consecutive user-sim failures with no success in between; "
            f"refusing to emit empty rollouts. Last error: {exc}"
        )


def post_sidecar_json(
    url: str,
    payload: dict[str, Any],
    *,
    timeout: float,
    attempts: int = 3,
    tag: str = "sidecar",
) -> dict[str, Any]:
    """POST to a CUE sidecar, retrying connection-level failures with backoff.

    Batch pre-decode holds one connection open for minutes, so a dropped connection (sidecar
    momentarily busy, or the server closing the socket) would otherwise throw away the whole
    chunk. Decode is idempotent, so replaying the request is safe; a sidecar that is genuinely
    down still fails after ``attempts``.
    """

    import urllib.request

    data = json.dumps(payload).encode("utf-8")
    last_exc: Exception | None = None
    for attempt in range(max(1, attempts)):
        req = urllib.request.Request(
            url, data=data, headers={"Content-Type": "application/json"}, method="POST",
        )
        try:
            with urllib.request.urlopen(req, timeout=timeout) as resp:  # noqa: S310
                return json.loads(resp.read().decode("utf-8"))
        except Exception as exc:  # noqa: BLE001
            last_exc = exc
            if attempt + 1 < max(1, attempts):
                warn(tag, f"sidecar POST failed (attempt {attempt + 1}): {exc}; retrying")
                time.sleep(5.0 * (attempt + 1))
    raise SidecarError(f"CUE sidecar POST {url} failed after {max(1, attempts)} attempts: {last_exc}")

# Serializes ONLY the decoder's GPU work (agent embed + command-block decode), which is
# not thread-safe. The (network) simulator call runs outside it so many episodes overlap there
# and vLLM can batch them -- see DecoderUserSim.next_turn.
_DECODER_GPU_LOCK = threading.Lock()

# Precomputed persona manuals: {f"{decoder_dir}::{arm}::{episode_id}" -> rendered command block}.
# A one-time batched decode (decode_manuals_batch) fills this so per-episode next_turn calls skip
# the (serialized, staggered) GPU decode and just run the parallel sim generation.
_MANUAL_CACHE: dict[str, str] = {}
# Decode settings + conditioning input behind each _MANUAL_CACHE entry. The cache key itself stays
# settings-free so next_turn can look a manual up without knowing how it was decoded;
# decode_manuals_batch only reuses an entry when both still match. Fingerprinting the input also
# makes reuse safe if two benchmarks sharing a sidecar ever mint the same arm::episode_id.
_MANUAL_SIG: dict[str, str] = {}
_MANUAL_LOCK = threading.Lock()


def _manual_cache_key(decoder_dir: str, key: str) -> str:
    return f"{decoder_dir}::{key}"


def _manual_input_fingerprint(value: Any) -> str:
    payload = json.dumps(value, sort_keys=True, default=str).encode("utf-8")
    return hashlib.sha1(payload, usedforsecurity=False).hexdigest()[:16]


# Generic/template command phrasing that makes decoded manuals read as assistant-like boilerplate
# rather than a real user's intents. Used only to down-rank candidates during reranking.
_GENERIC_COMMAND_PATTERNS = (
    "ask ", "request", "provide", "the user", "politely", "if the agent",
    "wants to", "would like", "needs to", "inquire", "state that", "make sure",
)


def _resolve_num_candidates(decode_mode: str, num_candidates: int) -> int:
    """How many manuals to decode per persona for ``decode_mode``."""

    mode = str(decode_mode or "greedy")
    n = max(1, int(num_candidates or 1))
    if mode == "greedy":
        return 1
    if mode == "diverse_slots":
        # One manual; ``num_candidates`` is per-slot sample count inside slot decode.
        return 1
    # sample: honor requested count (retrieval can run on the chosen candidate afterward)
    return n


def _slot_diverse_decode_kwargs(
    decode_mode: str,
    *,
    num_candidates: int,
    temperature: float,
    near_dup_jaccard: float = 0.5,
    noop_retries: int = 0,
    noop_temperature: float = 0.7,
) -> tuple[float, dict[str, Any]]:
    """Temperature + kwargs for ``generate_command_slot_blocks`` under ``decode_mode``."""

    mode = str(decode_mode or "greedy")
    if mode == "diverse_slots":
        n_per = max(2, int(num_candidates or 4))
        temp = float(temperature) if float(temperature) > 0 else 0.7
        return temp, {
            "num_candidates_per_slot": n_per,
            "near_dup_jaccard": float(near_dup_jaccard),
        }
    return float(temperature), {
        "num_candidates_per_slot": 1,
        "noop_retries": int(noop_retries),
        "noop_temperature": float(noop_temperature),
    }

def _manual_naturalness_score(block: dict[str, Any]) -> float:
    """Reference-free naturalness/diversity score for a parsed manual (``{commands, examples}``).

    Higher is better. Rewards command variety and concrete example utterances; penalizes
    over-questioning and generic/template command phrasing. Uses no human trajectory, so it is
    safe to apply at inference-time reranking."""

    commands = [str(c).strip() for c in (block.get("commands") or []) if str(c).strip()]
    examples = [str(x).strip() for x in (block.get("examples") or []) if str(x).strip()]
    if not commands:
        return float("-inf")
    score = 0.0
    score += min(len({c.casefold() for c in commands}), 6) * 0.5
    score += min(len(examples), 8) * 0.75
    if examples:
        q = sum(x.count("?") for x in examples)
        score -= (q / len(examples)) * 1.5
    generic = sum(
        1 for c in commands if any(p in c.casefold() for p in _GENERIC_COMMAND_PATTERNS)
    )
    score -= (generic / len(commands)) * 2.0
    return score


def _best_manual_block(candidates: list[str]) -> dict[str, Any]:
    """Parse each decoded candidate and return the highest-scoring ``{commands, examples}`` block.

    For a single candidate this is exactly ``parse_manual(text) or {empty}`` (greedy behavior)."""

    from cue_training.data.schema import parse_manual

    best_block: dict[str, Any] | None = None
    best_score = float("-inf")
    for text in candidates:
        block = parse_manual(text)
        if not block:
            continue
        s = _manual_naturalness_score(block)
        if s > best_score:
            best_score, best_block = s, block
    return best_block or {"commands": [], "examples": []}


def _best_dual_manual_pair(
    candidates: list[tuple[str, str, str]],
    *,
    cue_embedding: list[float] | None = None,
    decoder_dir: str | None = None,
    example_retrieval_enabled: bool = False,
    example_retrieval_k_sessions: int = 8,
    example_retrieval_n_general: int = 2,
    example_retrieval_n_specific: int = 2,
) -> str:
    """Pick the highest-scoring general/specific/style candidate and render it."""

    from cue_training.data.schema import merge_dual_manual_blocks, parse_manual
    from cue_training.refinement.simulator import render_dual_manual, render_manual

    best_score = float("-inf")
    best_general_text = ""
    best_specific_text = ""
    best_style_text = ""
    for general_text, specific_text, style_text in candidates:
        general = parse_manual(general_text) or {"commands": [], "examples": []}
        specific = parse_manual(specific_text) or {"commands": [], "examples": []}
        style = parse_manual(style_text) or {"commands": [], "examples": []}
        merged = merge_dual_manual_blocks(general, specific, style)
        score = _manual_naturalness_score(merged)
        if score > best_score:
            best_score = score
            best_general_text = general_text
            best_specific_text = specific_text
            best_style_text = style_text
    if example_retrieval_enabled and cue_embedding and decoder_dir:
        from cue_training.data.example_pool import get_pool_for_decoder_dir, inject_examples_into_dual_render

        pool = get_pool_for_decoder_dir(decoder_dir)
        if pool is None:
            warn(
                "cue:decode",
                f"example retrieval enabled but no example_pool under {decoder_dir}; "
                "run cue build-example-pool first",
            )
        else:
            rendered = inject_examples_into_dual_render(
                best_general_text,
                best_specific_text,
                cue_embedding,
                pool,
                k_sessions=example_retrieval_k_sessions,
                n_general=example_retrieval_n_general,
                n_specific=example_retrieval_n_specific,
            )
            style = parse_manual(best_style_text) or {"commands": [], "examples": []}
            style_render = render_dual_manual(None, None, style)
            return "\n\n".join(part for part in (rendered, style_render) if part)
    general = parse_manual(best_general_text) or {"commands": [], "examples": []}
    specific = parse_manual(best_specific_text) or {"commands": [], "examples": []}
    style = parse_manual(best_style_text) or {"commands": [], "examples": []}
    merged = merge_dual_manual_blocks(general, specific, style)
    return render_dual_manual(general, specific, style) or render_manual(merged)


def cached_manual(decoder_dir: str, key: str) -> str | None:
    return _MANUAL_CACHE.get(_manual_cache_key(decoder_dir, key))


def decode_manuals_batch(
    *,
    decoder_dir: str,
    items: list[tuple[str, list[float] | list[dict[str, Any]]]],
    device: str = "cuda",
    dtype: str = "bfloat16",
    max_new_tokens: int = 1024,
    temperature: float = 0.0,
    top_p: float = 1.0,
    batch_size: int = 16,
    num_candidates: int = 1,
    seed: int | None = None,
    decode_mode: str = "greedy",
    decode_slot_dedup_jaccard: float = 0.5,
    decode_noop_retries: int = 0,
    decode_noop_temperature: float = 0.7,
    example_retrieval_enabled: bool = False,
    example_retrieval_k_sessions: int = 8,
    example_retrieval_n_general: int = 2,
    example_retrieval_n_specific: int = 2,
    session_preprocess: str = "full",
) -> dict[str, str]:
    """Batch-decode persona manuals from trajectories or CUE embeddings, cache them, and
    return ``{key: command_block}``. Each GPU pass decodes up to ``batch_size`` sequences
    (personas x candidates) at once instead of a serialized per-episode decode -- the whole
    dataset's manuals are produced up front.

    ``decode_mode`` selects how a manual is chosen per persona (all reference-free):
    - ``greedy``: single greedy decode (prior behavior).
    - ``sample``: sample ``num_candidates`` (default 1); if >1, keep the most natural one,
      then optionally inject retrieved style examples.
    - ``diverse_slots``: for slot-decode models, sample ``num_candidates`` (≥2) commands
      per slot and keep the first that is below ``decode_slot_dedup_jaccard`` vs already-
      accepted commands in that head (threshold decorrelation; one manual per persona)."""
    import torch

    from cue_training.data.schema import (
        COMMAND_BLOCK_SENTINEL,
        GENERAL_COMMAND_SENTINEL,
        STYLE_COMMAND_SENTINEL,
        USER_SPECIFIC_SENTINEL,
        merge_dual_manual_blocks,
        parse_manual,
    )
    from cue_training.refinement.simulator import render_dual_manual, render_manual

    # Jobs sharing a decoder re-request the same manuals (cue general runs one sidecar for every
    # simulator, and a resume replays the whole arm), so reuse what this process already decoded
    # under identical settings instead of paying the GPU pass again.
    settings = repr(
        (
            dtype, max_new_tokens, temperature, top_p, num_candidates, seed, decode_mode,
            decode_slot_dedup_jaccard, decode_noop_retries, decode_noop_temperature,
            example_retrieval_enabled, example_retrieval_k_sessions,
            example_retrieval_n_general, example_retrieval_n_specific, session_preprocess,
        )
    )
    sigs = {key: f"{settings}::{_manual_input_fingerprint(value)}" for key, value in items}
    reused: dict[str, str] = {}
    with _MANUAL_LOCK:
        for key, _ in items:
            ck = _manual_cache_key(decoder_dir, key)
            if _MANUAL_CACHE.get(ck) and _MANUAL_SIG.get(ck) == sigs[key]:
                reused[key] = _MANUAL_CACHE[ck]
    if reused:
        log("cue-decoder", f"reusing {len(reused)}/{len(items)} cached manuals")
        items = [it for it in items if it[0] not in reused]
    if not items:
        return reused

    model = model_hub.get_decoder(decoder_dir=decoder_dir, device=device, dtype=dtype)
    decoder = model.decoder
    dev = decoder.lm.get_input_embeddings().weight.device
    dual_decode = bool(getattr(model, "dual_decode", False))
    expected = int(model.encoder.bottleneck_dim)

    n_cand = _resolve_num_candidates(decode_mode, num_candidates)
    slot_temp, slot_kwargs = _slot_diverse_decode_kwargs(
        decode_mode,
        num_candidates=num_candidates,
        temperature=temperature,
        near_dup_jaccard=decode_slot_dedup_jaccard,
        noop_retries=decode_noop_retries,
        noop_temperature=decode_noop_temperature,
    )
    gen_temp = (
        slot_temp
        if str(decode_mode or "") == "diverse_slots"
        else (temperature if temperature > 0 else (0.9 if n_cand > 1 else 0.0))
    )
    step = max(1, batch_size // n_cand)
    try:
        from tqdm.auto import tqdm

        bar = tqdm(total=len(items), desc="decode manuals", unit="manual")
    except Exception:  # noqa: BLE001 - tqdm optional; fall back to periodic log lines
        bar = None

    all_cands: dict[str, list[tuple[str, str, str]] | list[str]] = {}
    encoded_by_key: dict[str, list[float]] = {}
    with _DECODER_GPU_LOCK:
        for start in range(0, len(items), step):
            chunk = [it for it in items[start : start + step] if it[1] is not None]
            if not chunk:
                if bar is not None:
                    bar.update(len(items[start : start + step]))
                continue
            values = [value for _, value in chunk]
            trajectories = bool(
                values
                and isinstance(values[0], list)
                and (not values[0] or isinstance(values[0][0], dict))
            )
            if trajectories:
                from cue_training.infer.session_preprocess import preprocess_turns

                encode_values = [
                    preprocess_turns(turns, session_preprocess) for turns in values
                ]
                with torch.no_grad():
                    bn = model.encode(encode_values)
            else:
                bn = torch.tensor(values, dtype=torch.float32, device=dev)
            if bn.shape[-1] != expected:
                raise ValueError(
                    f"cue_embedding dim {bn.shape[-1]} != decoder cue_dim {expected}; re-export "
                    "embeddings with the encoder used to train this decoder."
                )
            for row, (key, _) in enumerate(chunk):
                encoded_by_key[key] = bn[row].detach().float().cpu().tolist()
            tiled = bn.repeat_interleave(n_cand, dim=0)
            if dual_decode:
                if bool(getattr(model, "command_slot_decode", False)):
                    general_blocks = model.generate_command_slot_blocks(
                        tiled,
                        head="general",
                        num_slots=int(getattr(model, "general_command_slots", 5)),
                        max_new_tokens=max_new_tokens,
                        temperature=gen_temp,
                        top_p=top_p,
                        **slot_kwargs,
                    )
                    specific_blocks = model.generate_command_slot_blocks(
                        tiled,
                        head="specific",
                        num_slots=int(
                            getattr(model, "user_specific_command_slots", 5)
                        ),
                        max_new_tokens=max_new_tokens,
                        temperature=gen_temp,
                        top_p=top_p,
                        **slot_kwargs,
                    )
                    general_decoded = [
                        json.dumps(block, ensure_ascii=False) for block in general_blocks
                    ]
                    specific_decoded = [
                        json.dumps(block, ensure_ascii=False) for block in specific_blocks
                    ]
                    style_slots = int(getattr(model, "style_command_slots", 0))
                    if style_slots > 0:
                        style_blocks = model.generate_command_slot_blocks(
                            tiled,
                            head="style",
                            num_slots=style_slots,
                            max_new_tokens=max_new_tokens,
                            temperature=gen_temp,
                            top_p=top_p,
                            **slot_kwargs,
                        )
                        style_decoded = [
                            json.dumps(block, ensure_ascii=False) for block in style_blocks
                        ]
                    else:
                        style_decoded = [
                            json.dumps({"commands": [], "examples": []})
                        ] * tiled.size(0)
                else:
                    general_sent = decoder.tokenizer(
                        GENERAL_COMMAND_SENTINEL,
                        add_special_tokens=False,
                        return_tensors="pt",
                    )["input_ids"][0].to(dev)
                    specific_sent = decoder.tokenizer(
                        USER_SPECIFIC_SENTINEL,
                        add_special_tokens=False,
                        return_tensors="pt",
                    )["input_ids"][0].to(dev)
                    if n_cand > 1 and seed is not None:
                        torch.manual_seed(seed + start)
                    general_decoded = decoder.generate(
                        bottleneck=tiled,
                        sentinel_ids=general_sent,
                        max_new_tokens=max_new_tokens,
                        temperature=gen_temp,
                        top_p=top_p,
                    )
                    if n_cand > 1 and seed is not None:
                        torch.manual_seed(seed + start + 1)
                    specific_decoded = decoder.generate(
                        bottleneck=tiled,
                        sentinel_ids=specific_sent,
                        max_new_tokens=max_new_tokens,
                        temperature=gen_temp,
                        top_p=top_p,
                    )
                    if float(getattr(model, "lambda_style_ce", 0.0)) > 0:
                        style_sent = decoder.tokenizer(
                            STYLE_COMMAND_SENTINEL,
                            add_special_tokens=False,
                            return_tensors="pt",
                        )["input_ids"][0].to(dev)
                        style_decoded = decoder.generate(
                            bottleneck=tiled,
                            sentinel_ids=style_sent,
                            max_new_tokens=max_new_tokens,
                            temperature=gen_temp,
                            top_p=top_p,
                        )
                    else:
                        style_decoded = [
                            json.dumps({"commands": [], "examples": []})
                        ] * tiled.size(0)
                for r, (key, _) in enumerate(chunk):
                    pairs: list[tuple[str, str, str]] = []
                    for c in range(n_cand):
                        idx = r * n_cand + c
                        pairs.append(
                            (
                                general_decoded[idx],
                                specific_decoded[idx],
                                style_decoded[idx],
                            )
                        )
                    all_cands[key] = pairs
            else:
                sentinel_ids = decoder.tokenizer(
                    COMMAND_BLOCK_SENTINEL, add_special_tokens=False, return_tensors="pt",
                )["input_ids"][0].to(dev)
                if n_cand > 1 and seed is not None:
                    torch.manual_seed(seed + start)
                decoded = decoder.generate(
                    bottleneck=tiled, sentinel_ids=sentinel_ids,
                    max_new_tokens=max_new_tokens, temperature=gen_temp, top_p=top_p,
                )
                for r, (key, _) in enumerate(chunk):
                    all_cands[key] = list(decoded[r * n_cand : (r + 1) * n_cand])
            if bar is not None:
                bar.update(len(items[start : start + step]))
    out = {}
    for key, value in all_cands.items():
        cue_emb = encoded_by_key.get(key)
        if dual_decode:
            out[key] = _best_dual_manual_pair(
                value,
                cue_embedding=cue_emb,
                decoder_dir=decoder_dir,
                example_retrieval_enabled=example_retrieval_enabled,
                example_retrieval_k_sessions=example_retrieval_k_sessions,
                example_retrieval_n_general=example_retrieval_n_general,
                example_retrieval_n_specific=example_retrieval_n_specific,
            )
        else:
            out[key] = render_manual(_best_manual_block(value))
    if bar is not None:
        bar.close()
    with _MANUAL_LOCK:
        for k, v in out.items():
            _MANUAL_CACHE[_manual_cache_key(decoder_dir, k)] = v
            _MANUAL_SIG[_manual_cache_key(decoder_dir, k)] = sigs[k]
    return {**reused, **out}


class UserSim:
    """Interface: reset once per episode, then produce user turns."""

    # When True, generation is offloaded to a batched server (vLLM) rather than a single
    # in-process GPU stream, so the sidecar does NOT serialize it behind the GPU lock.
    batched: bool = False
    # Optional per-turn provenance the sidecar returns and the harness records in rollout
    # metadata (e.g. the decoder's injected command block). None when not applicable.
    last_command: dict[str, Any] | None = None

    def reset(self, ctx: TurnContext) -> None:
        self.ctx = ctx

    def next_turn(self, history: list[Turn]) -> tuple[str, bool]:
        raise NotImplementedError


def _user_proxy_system(ctx: TurnContext) -> str:
    from cue_training.evaluation.common.mirror_prompts import user_proxy_system_prompt

    return user_proxy_system_prompt(task_description=ctx.task, domain=ctx.domain)


def _base_system(ctx: TurnContext) -> str:
    """Base user-sim system prompt: the harness's standard prompt when provided
    (for comparability across models), otherwise the CUE user-proxy prompt."""

    return ctx.system_prompt or _user_proxy_system(ctx)


class DecoderUserSim(UserSim):
    def __init__(self, ctx: TurnContext) -> None:
        self.ctx = ctx
        e = ctx.extra
        import torch

        from cue_training.data.schema import (
            COMMAND_BLOCK_SENTINEL,
            GENERAL_COMMAND_SENTINEL,
            STYLE_COMMAND_SENTINEL,
            USER_SPECIFIC_SENTINEL,
        )

        self.model = model_hub.get_decoder(
            decoder_dir=e["decoder_dir"], device=e.get("device", "cuda"),
            dtype=e.get("dtype", "bfloat16"),
        )
        self.decoder = self.model.decoder
        self.dual_decode = bool(getattr(self.model, "dual_decode", False))
        dev = self.decoder.lm.get_input_embeddings().weight.device
        self.sentinel_ids = self.decoder.tokenizer(
            COMMAND_BLOCK_SENTINEL, add_special_tokens=False, return_tensors="pt",
        )["input_ids"][0].to(dev)
        if self.dual_decode:
            self.general_sentinel_ids = self.decoder.tokenizer(
                GENERAL_COMMAND_SENTINEL, add_special_tokens=False, return_tensors="pt",
            )["input_ids"][0].to(dev)
            self.specific_sentinel_ids = self.decoder.tokenizer(
                USER_SPECIFIC_SENTINEL, add_special_tokens=False, return_tensors="pt",
            )["input_ids"][0].to(dev)
            self.style_sentinel_ids = self.decoder.tokenizer(
                STYLE_COMMAND_SENTINEL, add_special_tokens=False, return_tensors="pt",
            )["input_ids"][0].to(dev)
        self._torch = torch
        # Self-serialize the GPU decode via _DECODER_GPU_LOCK, so the sidecar must NOT also wrap this
        # sim in its coarse global lock (which would serialize the slow vLLM sim call too, starving
        # a high-fan-out run into client timeouts). batched=True tells the sidecar to skip that lock.
        self.batched = True
        # Decode once per episode: cache the session manual and reuse it on every turn.
        self._session_block: dict[str, Any] | None = None
        self._session_text = ""

    def next_turn(self, history: list[Turn]) -> tuple[str, bool]:
        torch = self._torch
        from cue_training.baselines.common.llm import chat
        from cue_training.baselines.common.prompts import user_turn_prompt
        from cue_training.refinement.simulator import render_manual
        from cue_training.evaluation.common.mirror_prompts import trim_control_message

        e = self.ctx.extra
        dev = self.decoder.lm.get_input_embeddings().weight.device
        max_new = int(e.get("command_max_new_tokens") or 1024)
        decode_temp = float(e.get("decode_temperature") or 0.0)
        decode_top_p = float(e.get("decode_top_p") or 1.0)
        decode_mode = str(e.get("decode_mode") or "greedy")
        n_req = int(e.get("decode_num_candidates") or 1)
        n_cand = _resolve_num_candidates(decode_mode, n_req)
        slot_temp, slot_kwargs = _slot_diverse_decode_kwargs(
            decode_mode,
            num_candidates=n_req,
            temperature=decode_temp,
            near_dup_jaccard=float(e.get("decode_slot_dedup_jaccard") or 0.5),
            noop_retries=int(e.get("decode_noop_retries") or 0),
            noop_temperature=float(e.get("decode_noop_temperature") or 0.7),
        )
        if self._session_block is None:
            # Reuse a precomputed manual (batched pre-decode) if available -> skip the GPU decode
            # entirely, so this call is just the (parallel) sim generation. Key: arm::episode_id.
            pre = e.get("command_block") or cached_manual(
                str(e.get("decoder_dir") or ""), f"{self.ctx.arm}::{self.ctx.episode_id}"
            )
            if pre is not None:
                self._session_block = {"precomputed": True}
                self._session_text = pre
                self.last_command = {"command_block": pre, "decoded": "(precomputed)"}
        if self._session_block is None:
            # GPU section only (not thread-safe): serialize it, but release before the (network) calls
            # so concurrent episodes overlap their batchable work instead of queueing on the GPU.
            with _DECODER_GPU_LOCK:
                cue = torch.tensor([self.ctx.cue_embedding], dtype=torch.float32, device=dev)
                expected = int(self.model.encoder.bottleneck_dim)
                if cue.shape[-1] != expected:
                    raise ValueError(
                        f"cue_embedding dim {cue.shape[-1]} != decoder cue_dim {expected}; "
                        "re-prep cases with the encoder embeddings used to train this decoder "
                        "(FORCE_PREP=1 EMBEDDINGS=...)."
                    )
                gen_temp = (
                    slot_temp
                    if decode_mode == "diverse_slots"
                    else (decode_temp if decode_temp > 0 else (0.9 if n_cand > 1 else 0.0))
                )
                if self.dual_decode:
                    if bool(getattr(self.model, "command_slot_decode", False)):
                        general_cands = []
                        specific_cands = []
                        style_cands = []
                        for _ in range(n_cand):
                            general = self.model.generate_command_slot_blocks(
                                cue,
                                head="general",
                                num_slots=int(
                                    getattr(self.model, "general_command_slots", 5)
                                ),
                                max_new_tokens=max_new,
                                temperature=gen_temp,
                                top_p=decode_top_p,
                                **slot_kwargs,
                            )[0]
                            specific = self.model.generate_command_slot_blocks(
                                cue,
                                head="specific",
                                num_slots=int(
                                    getattr(
                                        self.model,
                                        "user_specific_command_slots",
                                        5,
                                    )
                                ),
                                max_new_tokens=max_new,
                                temperature=gen_temp,
                                top_p=decode_top_p,
                                **slot_kwargs,
                            )[0]
                            style_slots = int(
                                getattr(self.model, "style_command_slots", 0)
                            )
                            style = (
                                self.model.generate_command_slot_blocks(
                                    cue,
                                    head="style",
                                    num_slots=style_slots,
                                    max_new_tokens=max_new,
                                    temperature=gen_temp,
                                    top_p=decode_top_p,
                                    **slot_kwargs,
                                )[0]
                                if style_slots > 0
                                else {"commands": [], "examples": []}
                            )
                            general_cands.append(
                                json.dumps(general, ensure_ascii=False)
                            )
                            specific_cands.append(
                                json.dumps(specific, ensure_ascii=False)
                            )
                            style_cands.append(json.dumps(style, ensure_ascii=False))
                    else:
                        general_cands = [
                            self.decoder.generate(
                                bottleneck=cue,
                                sentinel_ids=self.general_sentinel_ids,
                                max_new_tokens=max_new,
                                temperature=gen_temp,
                                top_p=decode_top_p,
                            )[0]
                            for _ in range(n_cand)
                        ]
                        specific_cands = [
                            self.decoder.generate(
                                bottleneck=cue,
                                sentinel_ids=self.specific_sentinel_ids,
                                max_new_tokens=max_new,
                                temperature=gen_temp,
                                top_p=decode_top_p,
                            )[0]
                            for _ in range(n_cand)
                        ]
                        style_cands = (
                            [
                                self.decoder.generate(
                                    bottleneck=cue,
                                    sentinel_ids=self.style_sentinel_ids,
                                    max_new_tokens=max_new,
                                    temperature=gen_temp,
                                    top_p=decode_top_p,
                                )[0]
                                for _ in range(n_cand)
                            ]
                            if float(getattr(self.model, "lambda_style_ce", 0.0)) > 0
                            else [
                                json.dumps({"commands": [], "examples": []})
                            ] * n_cand
                        )
                    self._session_text = _best_dual_manual_pair(
                        list(
                            zip(
                                general_cands,
                                specific_cands,
                                style_cands,
                                strict=True,
                            )
                        ),
                        cue_embedding=list(self.ctx.cue_embedding or []),
                        decoder_dir=str(e.get("decoder_dir") or ""),
                        example_retrieval_enabled=bool(e.get("example_retrieval_enabled")),
                        example_retrieval_k_sessions=int(e.get("example_retrieval_k_sessions") or 8),
                        example_retrieval_n_general=int(e.get("example_retrieval_n_general") or 2),
                        example_retrieval_n_specific=int(e.get("example_retrieval_n_specific") or 2),
                    )
                    self._session_block = {"dual_decode": True}
                    self.last_command = {
                        "command_block": self._session_text,
                        "decoded": (general_cands[0] or "")[:300],
                        "decoded_user_specific": (specific_cands[0] or "")[:300],
                        "decoded_style": (style_cands[0] or "")[:300],
                    }
                else:
                    cands = [
                        self.decoder.generate(
                            bottleneck=cue,
                            sentinel_ids=self.sentinel_ids,
                            max_new_tokens=max_new,
                            temperature=gen_temp,
                            top_p=decode_top_p,
                        )[0]
                        for _ in range(n_cand)
                    ]
                    self._session_block = _best_manual_block(cands)
                    self._session_text = render_manual(self._session_block)
                    self.last_command = {
                        "command_block": self._session_text,
                        "decoded": (cands[0] or "")[:600],
                    }
            if not self._session_text.strip():
                raw = (self.last_command or {}).get("decoded", "")
                warn(
                    "decoder",
                    f"empty decoded manual (episode={self.ctx.episode_id}); "
                    f"decoded_raw={raw[:200]!r}",
                )
        command_text = self._session_text
        if e.get("simarena_native_prompt"):
            from cue_training.baselines.realusersim.simulator import _ANTI_NORMALIZATION
            from cue_training.evaluation.simulatorarena.prompts import native_user_prompt

            # Carry anti-normalization like the non-native branch below, and like RealUserSim
            # (whose augmentation() brings it into the shared prompt) -- without it the sim
            # drifts into the assistant's register instead of the persona's.
            native_profile = command_text.strip()
            if native_profile:
                native_profile = f"{native_profile}\n{_ANTI_NORMALIZATION}"
            messages = [{
                "role": "user",
                "content": native_user_prompt(
                    domain=str(self.ctx.domain or "writing"),
                    history=list(history),
                    user_profile=native_profile,
                    task=self.ctx.task,
                    document_type=e.get("document_type"),
                    intent=e.get("intent"),
                    background=e.get("background"),
                    math_problem=e.get("math_problem"),
                ),
            }]
        else:
            # Same framing as RealUserSim / PromptUserSimulator: baselines user_system_prompt +
            # persona augmentation + user_turn_prompt (not MirrorBench proxy or SimArena CoT).
            from cue_training.baselines.common.prompts import user_system_prompt
            from cue_training.baselines.realusersim.simulator import _ANTI_NORMALIZATION

            aug = ""
            if command_text.strip():
                aug = (
                    "You are imitating ONE specific real user with this persona.\n"
                    f"{command_text.strip()}\n"
                    f"{_ANTI_NORMALIZATION}"
                )
            system = user_system_prompt(
                self.ctx.task,
                {"domain": self.ctx.domain},
                aug,
            )
            if self.ctx.system_prompt:
                system = f"{self.ctx.system_prompt.strip()}\n\n{system}"
            messages = [
                {"role": "system", "content": system},
                {"role": "user", "content": user_turn_prompt(self.ctx.task, list(history))},
            ]
        max_tokens = int(e.get("sim_max_tokens", 512))
        sim_model = e.get("sim_model", "gpt-5.4-mini")
        messages = _fit_sim_messages(
            messages, extra=e, max_tokens=max_tokens, model=str(sim_model)
        )
        # At high fan-out one slow simulator response would otherwise abort the whole run, so
        # retry the transient classes with backoff (same shape as the sidecar retry below).
        attempts = max(1, int(e.get("sim_retries", 3)))
        raw = ""
        for attempt in range(attempts):
            try:
                raw = chat(
                    messages, model=sim_model,
                    api_key_env=e.get("sim_api_key_env", "OPENAI_API_KEY"),
                    api_base=e.get("sim_api_base"),
                    temperature=float(e.get("sim_temperature", 0.7)),
                    max_tokens=max_tokens,
                    top_p=float(e.get("sim_top_p", 0.95)),
                    # Bound the steered-sim call so a dead/hung endpoint fails fast (surfacing the real
                    # error) instead of blocking until the caller's sidecar_timeout. Tune via sim_timeout.
                    timeout=float(e.get("sim_timeout", 120.0)),
                )
                break
            except Exception as exc:  # noqa: BLE001
                if attempt + 1 < attempts and _is_transient_sim_error(exc):
                    warn("cue-decoder", f"sim call error (attempt {attempt + 1}): {exc}; retrying")
                    time.sleep(2.0 * (attempt + 1))
                    continue
                # Decoder is CUE-only: fail loudly so a broken decoder/sim aborts the run instead of
                # emitting a silent ###STOP### (see SidecarError).
                raise SidecarError(f"CUE decoder sim call failed (episode={self.ctx.episode_id}): {exc}") from exc
        if e.get("simarena_native_prompt"):
            from cue_training.evaluation.simulatorarena.prompts import parse_native_user_output

            msg, stop = parse_native_user_output(raw)
            return msg, stop
        # Honor mid-turn ### as stop (same as BaselineUserSim); strip think/control wrappers first.
        msg, stop = trim_control_message(raw.strip(), return_stop=True)
        msg = msg or ""
        if not msg and not stop:
            # Empty after trimming = the steered model emitted only control tokens (no ###).
            # Log the raw output + manual size so an immediate empty end is diagnosable.
            warn(
                "decoder",
                f"empty user turn (episode={self.ctx.episode_id}, turn={len(history)}); "
                f"manual_chars={len(self._session_text)} raw_sim_output={raw[:200]!r}",
            )
        return msg, bool(stop or not msg)


class BaselineUserSim(UserSim):
    # attribute each baseline uses to hold its per-episode persona (for injection)
    _PERSONA_ATTR = {
        "realusersim": "_profile",
        "realusersim_paired_noex": "_profile",
        "usp": "_profile_text",
        "ppol": "_persona",
    }

    def __init__(self, ctx: TurnContext) -> None:
        self.ctx = ctx
        e = ctx.extra
        self.name = e["name"]
        # Forward USP's optional vLLM-serving config so the published HF USP
        # model is batched by a local vLLM server.
        sim_extra = {
            k: e[k]
            for k in (
                "usp_vllm_base_url", "usp_vllm_model", "usp_vllm_api_key_env",
                "usp_official_vllm_base_url", "usp_official_vllm_model",
                "usp_official_vllm_api_key_env",
                "userlm_vllm_base_url", "userlm_vllm_model", "userlm_vllm_api_key_env",
            )
            if e.get(k) is not None
        }
        self.simulator = model_hub.get_baseline_simulator(
            name=self.name, output_dir=e["output_dir"], artifacts_dir=e.get("artifacts_dir"),
            sim_model=e.get("sim_model", "gpt-5.4-mini"), device=e.get("device", "cuda"),
            sim_api_base=e.get("sim_api_base"), sim_api_key_env=e.get("sim_api_key_env", "OPENAI_API_KEY"),
            extra=sim_extra,
        )
        # A vLLM-served sim batches on the server, so the sidecar can skip its GPU lock.
        self.batched = bool(getattr(self.simulator, "batched", False))

    def reset(self, ctx: TurnContext) -> None:
        self.ctx = ctx
        metadata = {
            "domain": ctx.domain,
            "episode_id": ctx.episode_id,
            "arm": ctx.arm,
            "cue_embedding": ctx.cue_embedding,
        }
        start = getattr(self.simulator, "_on_episode_start", None)
        if callable(start):
            try:
                start(ctx.task, metadata)
            except Exception:  # noqa: BLE001
                pass
        # Inject the paired/sample_shuffled persona over native pool sampling where supported.
        if ctx.persona is not None and self.name in self._PERSONA_ATTR:
            setattr(self.simulator, self._PERSONA_ATTR[self.name], ctx.persona)
        # Prepend the harness's standard user-sim prompt for prompt-based baselines
        # (additive: keeps the baseline's own prompt/persona). Trained baselines with a
        # fixed system-prompt contract (e.g. UserLM's system=task, USP's profile-only
        # chat template) lack this attribute and are untouched — USP instead folds
        # ``ctx.task`` into its profile slot inside ``USPOfficialVLLMSimulator``.
        if ctx.system_prompt and hasattr(self.simulator, "extra_system_prompt"):
            self.simulator.extra_system_prompt = ctx.system_prompt
        # Run prompt-based baselines through SimulatorArena's native CoT template too, so CUE
        # and the baselines share one format and stay comparable (each baseline's persona is
        # still injected, appended last by native_user_prompt). Trained baselines with a fixed
        # system-prompt contract (UserLM, USP) lack the hook and are skipped.
        if ctx.extra.get("simarena_native_prompt") and hasattr(self.simulator, "prompt_builder"):
            self.simulator.prompt_builder = self._native_messages

    def _native_messages(
        self, task: str, history: list[Turn], persona: str
    ) -> list[dict[str, str]]:
        from cue_training.evaluation.simulatorarena.prompts import native_user_prompt

        e = self.ctx.extra
        return [{
            "role": "user",
            "content": native_user_prompt(
                domain=str(self.ctx.domain or "writing"),
                history=list(history),
                user_profile=persona,
                task=task,
                document_type=e.get("document_type"),
                intent=e.get("intent"),
                background=e.get("background"),
                math_problem=e.get("math_problem"),
            ),
        }]

    def next_turn(self, history: list[Turn]) -> tuple[str, bool]:
        from cue_training.evaluation.common.rollout import clean_user_message

        metadata = {
            "domain": self.ctx.domain,
            "episode_id": self.ctx.episode_id,
            "arm": self.ctx.arm,
            "cue_embedding": self.ctx.cue_embedding,
        }
        attempts = max(1, int(self.ctx.extra.get("sim_retries", 3)))
        for attempt in range(attempts):
            try:
                text = self.simulator.next_turn(self.ctx.task, list(history), metadata)
                break
            except Exception as exc:  # noqa: BLE001
                if attempt + 1 < attempts and _is_transient_sim_error(exc):
                    time.sleep(2.0 * (attempt + 1))
                    continue
                warn(self.name, f"next_turn failed: {exc}")
                _note_baseline_sim_failure(self.name, exc)
                return "", True
        _note_baseline_sim_success()
        self.last_command = getattr(self.simulator, "last_command", None)
        # Parse a ``###``-style stop marker (and strip role prefixes / wrapping quotes).
        # UserLM terminates via ``<|endconversation|>`` (``is_done``); other baselines
        # defer natural-end to ``_StopGuard``. First-user-turn end is blocked in ``_StopGuard``.
        if getattr(self.simulator, "prompt_builder", None) is not None:
            from cue_training.evaluation.simulatorarena.prompts import parse_native_user_output

            return parse_native_user_output(text)
        msg, stop = clean_user_message(text, return_stop=True)
        userlm_done = False
        if self.name == "userlm" and hasattr(self.simulator, "is_done"):
            try:
                userlm_done = bool(self.simulator.is_done(list(history)))
            except Exception:  # noqa: BLE001
                userlm_done = False
        return msg, bool(stop or not msg or userlm_done)


class BaseModelUserSim(UserSim):
    """Plain user LM, no persona/CUE conditioning (floor).

    Backends: ``base_api`` (provider API); ``base_local`` either loads the HF model in
    process (GPU) or, when ``extra['vllm_base_url']`` is set, is served by a local vLLM
    OpenAI server (continuous batching -> high concurrency) via litellm's ``hosted_vllm``.
    """

    def __init__(self, ctx: TurnContext) -> None:
        self.ctx = ctx
        e = ctx.extra
        self.vllm_base_url = e.get("vllm_base_url") if ctx.variant == "base_local" else None
        self.batched = bool(self.vllm_base_url)  # vLLM batches -> sidecar skips the GPU lock
        # HF in-process only for base_local WITHOUT a vLLM endpoint.
        self.local = ctx.variant == "base_local" and not self.vllm_base_url
        if self.local:
            self.handle = model_hub.get_base_local(
                hf_path=e["hf_path"], device=e.get("device", "cuda"), dtype=e.get("dtype", "bfloat16")
            )

    def _messages(self, history: list[Turn]) -> list[dict[str, str]]:
        from cue_training.baselines.common.prompts import user_system_prompt, user_turn_prompt

        metadata = {"domain": self.ctx.domain}
        system = self.ctx.system_prompt or user_system_prompt(self.ctx.task, metadata)
        return [
            {"role": "system", "content": system},
            {"role": "user", "content": user_turn_prompt(self.ctx.task, list(history))},
        ]

    def next_turn(self, history: list[Turn]) -> tuple[str, bool]:
        e = self.ctx.extra
        messages = self._messages(history)
        if not self.local:
            from cue_training.baselines.common.llm import chat

            if self.vllm_base_url:  # base_local served by a local vLLM OpenAI endpoint
                served = e.get("hf_path", "meta-llama/Llama-3.1-8B-Instruct")
                model = f"hosted_vllm/{served}"
                api_base, api_key_env = self.vllm_base_url, e.get("vllm_api_key_env", "HOSTED_VLLM_API_KEY")
            else:  # base_api (provider API)
                model = e.get("sim_model", "gpt-5.4-mini")
                api_base, api_key_env = e.get("sim_api_base"), e.get("sim_api_key_env", "OPENAI_API_KEY")
            try:
                max_tokens = int(e.get("sim_max_tokens", e.get("max_tokens", 512)))
                temperature = float(e.get("sim_temperature", e.get("temperature", 0.7)))
                top_p = float(e.get("sim_top_p", e.get("top_p", 0.95)))
                # Mirror decoder fitting for local vLLM; API sims leave sim_max_input_tokens unset.
                fit_extra = dict(e)
                if self.vllm_base_url:
                    fit_extra.setdefault("sim_api_base", self.vllm_base_url)
                messages = _fit_sim_messages(
                    messages, extra=fit_extra, max_tokens=max_tokens, model=model
                )
                text = chat(
                    messages, model=model, api_key_env=api_key_env, api_base=api_base,
                    temperature=temperature, max_tokens=max_tokens, top_p=top_p,
                )
            except Exception as exc:  # noqa: BLE001
                warn("base-model", f"chat failed ({model}): {exc}")
                return "", True
            text = (text or "").strip()
            return text, not text
        # local HF
        import torch

        h = self.handle
        tok, lm, dev = h["tokenizer"], h["lm"], h["device"]
        prompt = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        enc = tok(prompt, return_tensors="pt", add_special_tokens=False).to(dev)
        temperature = float(e.get("sim_temperature", e.get("temperature", 0.7)))
        max_new = int(e.get("sim_max_tokens", e.get("max_new_tokens", e.get("max_tokens", 128))))
        with torch.no_grad():
            out = lm.generate(
                **enc, max_new_tokens=max_new,
                do_sample=temperature > 0, temperature=temperature if temperature > 0 else None,
                top_p=float(e.get("sim_top_p", e.get("top_p", 0.95))),
                pad_token_id=tok.pad_token_id, eos_token_id=tok.eos_token_id,
            )
        text = tok.decode(out[0][enc["input_ids"].shape[1]:], skip_special_tokens=True).strip()
        return text, not text


# Variants that load a local GPU model; when a sidecar_url is provided these are served
# remotely so the harness driver process stays CPU-only. API variants run in-process.
_GPU_VARIANTS = {"decoder", "base_local"}
_GPU_BASELINES = {"userlm", "usp"}


def _needs_gpu(variant: str) -> bool:
    if variant in _GPU_VARIANTS:
        return True
    if variant.startswith("baseline:"):
        return variant.split(":", 1)[1] in _GPU_BASELINES
    return False


def _policy_vllm_ready(variant: str, extra: dict) -> bool:
    name = variant.split(":", 1)[1] if variant.startswith("baseline:") else ""
    if name == "userlm":
        return bool(extra.get("userlm_vllm_base_url"))
    if name == "usp":
        return bool(extra.get("usp_vllm_base_url"))
    return False


class RemoteUserSim(UserSim):
    """Produce the next user turn by calling the shared CUE sidecar (`/next_turn`).

    Sends the full per-episode conditioning each call so the sidecar (on a GPU node)
    can build/cache the real sim; keeps the caller CPU-only. Used automatically by
    ``build_user_sim`` for GPU variants when ``ctx.extra['sidecar_url']`` is set.
    """

    def __init__(self, ctx: TurnContext) -> None:
        self.ctx = ctx
        self._url = str(ctx.extra["sidecar_url"]).rstrip("/")
        self._timeout = float(ctx.extra.get("sidecar_timeout", 600.0))

    def next_turn(self, history: list[Turn]) -> tuple[str, bool]:
        import json
        import time as _time
        import urllib.request

        payload = {
            "variant": self.ctx.variant, "arm": self.ctx.arm, "episode_id": self.ctx.episode_id,
            "task": self.ctx.task, "domain": self.ctx.domain,
            "cue_embedding": self.ctx.cue_embedding, "persona": self.ctx.persona,
            # Forward the harness's standard system prompt (e.g. tau2's, which carries the task
            # instructions) so the sidecar builds on it instead of the generic CUE user-proxy
            # fallback -- otherwise the injected command block has no task to ground on.
            "system_prompt": self.ctx.system_prompt,
            "extra": self.ctx.extra,
            "history": list(history),
        }
        data = json.dumps(payload).encode("utf-8")
        strict = self.ctx.variant == "decoder"  # CUE decoder: fail loudly, don't emit silent STOP
        # Retry transient connection errors (reset/refused while the sidecar is momentarily busy or
        # restarting) with backoff before giving up; a genuinely dead sidecar still fails after these.
        attempts = int(self.ctx.extra.get("sidecar_retries", 3))
        last_exc: Exception | None = None
        body: dict[str, Any] | None = None
        for attempt in range(max(1, attempts)):
            req = urllib.request.Request(
                f"{self._url}/next_turn", data=data,
                headers={"Content-Type": "application/json"}, method="POST",
            )
            try:
                with urllib.request.urlopen(req, timeout=self._timeout) as resp:  # noqa: S310
                    body = json.loads(resp.read().decode("utf-8"))
                break
            except Exception as exc:  # noqa: BLE001
                last_exc = exc
                if attempt + 1 < max(1, attempts):
                    warn("remote-sim", f"sidecar call error (attempt {attempt + 1}): {exc}; retrying")
                    _time.sleep(2.0 * (attempt + 1))
        if body is None:
            if strict:
                msg = f"CUE sidecar call failed (episode={self.ctx.episode_id}): {last_exc}"
                _record_sidecar_error(msg)
                raise SidecarError(msg) from last_exc
            warn("remote-sim", f"sidecar call failed: {last_exc}")
            return "", True
        err = body.get("error")
        if err and strict:
            msg = f"CUE sidecar returned error (episode={self.ctx.episode_id}): {err}"
            _record_sidecar_error(msg)
            raise SidecarError(msg)
        self.last_command = body.get("command")  # per-turn provenance from the sidecar
        return (body.get("text") or "").strip(), bool(body.get("done"))


class _StopGuard(UserSim):
    """Driver-side termination backstop wrapped around any ``UserSim``.

    CUE local sims never learned to emit the ``###`` stop marker, so a decoder
    sim can degenerate into pinging a closing line ("Goodbye.") until the step cap.
    This ends the episode when either (a) the new user turn repeats a recent user
    turn verbatim (a degeneration loop), or (b) an LLM judge decides the
    conversation has reached a natural end. It only ever *adds* a stop -- an inner
    ``done`` (or empty) turn is passed through unchanged. Both checks are model-
    agnostic, so every arm terminates the same way. Configure via ``ctx.extra``:
    ``stop_repeat_threshold`` (consecutive identical user turns, default 2; <2
    disables), ``llm_end_check`` (default False — Tau2/user-sim prompts already
    teach ``###STOP###``, and the LLM judge was ending on bare confirmations),
    ``end_check_model`` (default ``sim_model``), ``end_check_max_words`` (LLM
    prefilter, default 12).
    """

    def __init__(self, inner: UserSim, ctx: TurnContext) -> None:
        self.inner = inner
        self.ctx = ctx
        e = ctx.extra
        self.repeat_threshold = int(e.get("stop_repeat_threshold", 2))
        self.llm_end_check = bool(e.get("llm_end_check", False))
        self.end_check_model = e.get("end_check_model") or e.get("sim_model", "gpt-5.4-mini")
        self.end_check_max_words = int(e.get("end_check_max_words", 12))
        self.batched = getattr(inner, "batched", False)

    def reset(self, ctx: TurnContext) -> None:
        self.ctx = ctx
        self.inner.reset(ctx)

    @staticmethod
    def _norm(text: str) -> str:
        return " ".join(text.strip().lower().split()).rstrip(".!?")

    def _is_repeat(self, history: list[Turn], text: str) -> bool:
        if self.repeat_threshold < 2:
            return False
        prev_users = [t["content"] for t in history if t.get("role") == "user"]
        if len(prev_users) < self.repeat_threshold - 1:
            return False
        target = self._norm(text)
        if not target:
            return False
        tail = prev_users[-(self.repeat_threshold - 1):]
        return all(self._norm(u) == target for u in tail)

    def _looks_ended(self, history: list[Turn], text: str) -> bool:
        # Cheap prefilter: only consult the judge on short, non-question turns
        # (a substantive ask or a question is clearly not a sign-off).
        if not self.llm_end_check:
            return False
        if text.rstrip().endswith("?") or len(text.split()) > self.end_check_max_words:
            return False

        from cue_training.baselines.common.llm import chat

        turns = [*history[-6:], {"role": "user", "content": text}]
        transcript = "\n".join(f"{t['role'].upper()}: {t['content']}" for t in turns)
        messages = [
            {
                "role": "system",
                "content": (
                    "You judge whether a user<->agent conversation has reached a "
                    "natural end: the user has nothing further to ask and is signing "
                    "off. Answer with only YES or NO."
                ),
            },
            {
                "role": "user",
                "content": (
                    f"Conversation (most recent turns):\n{transcript}\n\n"
                    "Has the conversation reached a natural end? Answer YES or NO."
                ),
            },
        ]
        e = self.ctx.extra
        from cue_training.baselines.common.llm import _has_thinking_budget, _needs_output_headroom

        # Reasoning (gpt-5*/o*) and Gemini thinking models spend hidden tokens from the same
        # output budget; a 3-token cap truncates before any YES/NO (finish_reason=length,
        # empty content). Give them room; keep the tiny cap for plain chat models.
        max_toks = int(e.get("end_check_max_tokens", 0)) or (
            256 if _needs_output_headroom(self.end_check_model) else 3
        )
        chat_kwargs: dict = {}
        # Gemini 3 Flash defaults to high thinking; starve it for a YES/NO check.
        if _has_thinking_budget(self.end_check_model):
            chat_kwargs["reasoning_effort"] = "none"
        try:
            out = chat(
                messages, model=self.end_check_model,
                api_key_env=e.get("sim_api_key_env", "OPENAI_API_KEY"),
                api_base=e.get("sim_api_base"), temperature=0.0, max_tokens=max_toks,
                **chat_kwargs,
            )
        except Exception as exc:  # noqa: BLE001
            warn("stop-guard", f"end-check failed ({self.end_check_model}): {exc}")
            return False
        return (out or "").strip().upper().startswith("YES")

    def next_turn(self, history: list[Turn]) -> tuple[str, bool]:
        text, done = self.inner.next_turn(history)
        self.last_command = getattr(self.inner, "last_command", None)  # surface inner provenance
        # First user turn: never terminate (UserLM / tau2 agent-first collapse prevention).
        # Paper UserLM still uses <|endconversation|> after turn 1; honor that via ``done``.
        prior_users = sum(1 for t in history if t.get("role") == "user")
        if prior_users == 0:
            if not text:
                return text, True
            return text, False
        if done or not text:
            return text, done
        if self._is_repeat(history, text) or self._looks_ended(history, text):
            return text, True
        return text, False


def build_user_sim(ctx: TurnContext) -> UserSim:
    """Construct the UserSim for ``ctx.variant`` and reset it for the episode.

    When ``ctx.extra['sidecar_url']`` is set and the variant needs a GPU, returns a
    ``RemoteUserSim`` that offloads generation to the shared sidecar. Non-remote sims
    are wrapped in ``_StopGuard`` (remote sims are guarded on the sidecar, where their
    real sim is built through this same function, so the guard runs exactly once).
    Policy baselines / base_local with a live ``*_vllm_base_url`` run in-process.
    """

    variant = ctx.variant
    sim: UserSim
    use_remote = (
        bool(ctx.extra.get("sidecar_url"))
        and _needs_gpu(variant)
        and not _policy_vllm_ready(variant, ctx.extra)
        and not (variant == "base_local" and ctx.extra.get("vllm_base_url"))
    )
    if use_remote:
        sim = RemoteUserSim(ctx)
        sim.reset(ctx)
        return sim
    if variant == "decoder":
        sim = DecoderUserSim(ctx)
    elif variant.startswith("baseline:"):
        ctx.extra.setdefault("name", variant.split(":", 1)[1])
        sim = BaselineUserSim(ctx)
    elif variant in ("base_api", "base_local"):
        sim = BaseModelUserSim(ctx)
    else:
        raise ValueError(f"Unknown user-sim variant: {variant!r}")
    sim.reset(ctx)
    return _StopGuard(sim, ctx)
