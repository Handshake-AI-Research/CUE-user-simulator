"""Extract Executable Persona Manuals + demographics (RealUserSim paper).

Paper training (arXiv:2605.20204 §3.1 / App. A): curate WildChat-4.8M → 21,637
English GPT-4o trajectories / 7,311 users, then GPT-4o builds one Executable
Persona Manual per user (~7,275 profiles). No model weights are updated.

Eval (§4.2 τ-bench): randomly sample a grounded WildChat profile per task
(``as_is`` pool sampling) — not episode-paired τ-USI extraction.
"""

from __future__ import annotations

import json
import os
import re
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

from cue_training.baselines.common.config import TrainConfig
from cue_training.baselines.common.data import training_conversations
from cue_training.baselines.common.jsonl import read_jsonl, write_jsonl
from cue_training.baselines.common.llm import achat, chat, parse_json_block
from cue_training.baselines.common.prompts import transcript_text
from cue_training.baselines.realusersim.format_persona import DEMOGRAPHIC_FIELDS, command_example_pairs
from cue_training.runlog.log import log, warn

PROFILES_FILE = "profiles.jsonl"
DEFAULT_PROFILER_MODEL = "gpt-4o"

_EXTRACTION_PROMPT = (
    "You analyze a real human user's messages in a conversation with an AI assistant and "
    "produce an Executable Persona Manual that another LLM can follow to imitate this user.\n"
    "Return STRICT JSON with keys:\n"
    '  "demographics": {\n'
    '    for each of age, gender, education, occupation, location, income, '
    "marital_status, nationality:\n"
    '      {"value": str|null, "source": "extracted"|"inferred"|null, "confidence": float|null}\n'
    "  },\n"
    '  "additional_background": free-text context from the conversation history or null,\n'
    '  "command_examples": [up to 15 objects {"command": str, "examples": [str, ...]}].\n'
    "Rules (match RealUserSim):\n"
    "- Each command is a direct imperative instruction for an LLM simulator covering "
    "recurring communication patterns across eight dimensions: capitalization, punctuation, "
    "message length, formality, filler/hesitation, repetition, multi-turn behavior, and "
    "requests for alternatives.\n"
    "- Each example MUST be quoted verbatim from the user's actual messages (do not rewrite, "
    "paraphrase, or delexicalize).\n"
    "- Prefer explicit self-disclosures for demographics (source=extracted); otherwise infer "
    "from conversational cues (source=inferred) and set a confidence in [0,1]. Use null when "
    "unknown.\n"
    "- Focus on HOW the user writes, not the task content of this one conversation.\n"
)

# Domain / slot nouns that make commands unlearnable from CUE alone (task leakage).
_DOMAIN_TOKEN_RE = re.compile(
    r"\b("
    r"flight|flights|airline|airport|hotel|hotels|restaurant|restaurants|booking|bookings|"
    r"reservation|reservations|taxi|uber|train|trains|bus|buses|movie|movies|theater|"
    r"theatre|ticket|tickets|insurance|policy|coverage|amenity|amenities|wifi|check[- ]?in|"
    r"check[- ]?out|itinerary|destination|departure|arrival"
    r")\b",
    re.I,
)
_BREVITY_RE = re.compile(r"\b(brief|short|terse|concise|succinct)\b", re.I)
_DESCRIPTIVE_PREFIX_RE = re.compile(
    r"^(the user\s+)?(uses|often|frequently|may|can|prefers|generally|sometimes|tends to|is)\b",
    re.I,
)


def persistent_when_prompt(*, n_persistent: int, m_when: int) -> str:
    """data-annotation / decoder-refinement prompt: N persistent style rules + M conditional when-rules."""

    return (
        "You distill an EXECUTABLE persona manual for a user simulator. "
        "Return STRICT JSON with keys:\n"
        '  "demographics": {"expertise": str, "tone": str, "verbosity": str, "background": str},\n'
        f'  "persistent_commands": [up to {n_persistent} second-person imperative rules for '
        "HOW this user usually writes],\n"
        f'  "when_commands": [up to {m_when} second-person imperative rules that MUST start with '
        '"When " and say how style CHANGES after dialogue events '
        "(clarification, correction, pushback, failure, frustration)],\n"
        '  "examples": [2-4 short user messages that show STYLE only].\n'
        "HARD RULES:\n"
        "- Write commands as instructions to the simulator: start with a verb "
        "(Use / Prefer / Omit / Answer / Keep / Write / Ask / Confirm / Restate...). "
        "Do NOT write descriptive third-person ('Uses…', 'Often…', 'The user…').\n"
        "- Style/behavior only: terseness, casing, punctuation, typos, politeness, directness, "
        "fragmentation, incremental constraint-giving. NEVER mention domains or slots "
        "(no hotel/flight/restaurant/booking/reservation/ticket/movie/taxi/train/wifi/dates/"
        "prices/locations/ratings/amenities).\n"
        "- At most ONE of the persistent commands may be about being brief/short/terse/concise; "
        "the rest must name DISTINCT observable habits.\n"
        "- When-commands must be task-agnostic triggers "
        "('When the assistant asks a clarifying question, …') not content triggers.\n"
        "- Examples: rewrite real turns into short style exemplars with placeholders if needed "
        "(e.g. 'yes the earlier one', 'no make it later', 'ok') — strip concrete entities.\n"
        "- Demographics.background must stay domain-agnostic (no travel/food/movies).\n"
        "GOOD: 'Use short lowercase fragments without end punctuation.'\n"
        "GOOD: 'When asked to clarify, answer with only the missing detail.'\n"
        "BAD: 'Uses short, direct requests.'\n"
        "BAD: 'When booking, ask for the hotel wifi.'\n"
    )


def contrastive_rerank_prompt(*, n_persistent: int, m_when: int) -> str:
    """Select discriminative commands from candidates using target/positive/negative evidence."""

    return (
        "You rerank candidate persona commands. Return STRICT JSON with only:\n"
        f'  "persistent_commands": [up to {n_persistent} exact candidate strings],\n'
        f'  "when_commands": [up to {m_when} exact candidate strings].\n'
        "Keep a command only when it is supported by the TARGET and POSITIVE evidence and "
        "distinguishes them from the NEGATIVE contrast evidence. Prefer observable, specific "
        "style habits over generic rules such as being concise, direct, or polite. Do not "
        "rewrite or invent commands; choose exact strings from the candidate lists. The target "
        "examples are evidence only and must never be emitted as examples."
    )


# Back-compat alias for callers that still use the private name.
_persistent_when_prompt = persistent_when_prompt


def _user_only(conversation: list[dict[str, str]]) -> str:
    users = [t.get("content", "") for t in conversation if t.get("role") == "user"]
    return transcript_text([{"role": "user", "content": u} for u in users])


def normalize_profile(parsed: dict[str, Any]) -> dict[str, Any] | None:
    """Canonicalize paper or legacy profile JSON into a RealUserSim profile dict."""

    if not isinstance(parsed, dict):
        return None
    pairs = command_example_pairs(parsed)
    ling = str(parsed.get("linguistic_profile") or parsed.get("manual_sections") or "").strip()
    if not pairs and not (
        parsed.get("commands") or parsed.get("examples") or ling
    ):
        return None
    demo_in = parsed.get("demographics") if isinstance(parsed.get("demographics"), dict) else {}
    demographics: dict[str, Any] = {}
    for field in DEMOGRAPHIC_FIELDS:
        info = demo_in.get(field)
        if isinstance(info, dict):
            demographics[field] = {
                "value": info.get("value"),
                "source": info.get("source"),
                "confidence": info.get("confidence"),
            }
        elif info is not None and str(info).strip():
            demographics[field] = {"value": str(info).strip(), "source": "inferred", "confidence": None}
        else:
            demographics[field] = {"value": None, "source": None, "confidence": None}
    # Keep lean keys if present (older CUE extracts) without inventing paper fields.
    for k, v in demo_in.items():
        if k in demographics:
            continue
        if isinstance(v, dict):
            demographics[k] = v
        elif v is not None and str(v).strip():
            demographics[k] = str(v).strip()
    command_examples = [
        {"command": cmd, "examples": list(ex)} for cmd, ex in pairs
    ]
    # Flat mirrors for callers that still read commands/examples.
    commands = [p["command"] for p in command_examples]
    examples: list[str] = []
    for p in command_examples:
        for e in p["examples"]:
            if e not in examples:
                examples.append(e)
    out: dict[str, Any] = {
        "demographics": demographics,
        "additional_background": (
            str(parsed.get("additional_background")).strip()
            if parsed.get("additional_background")
            else None
        ),
        "command_examples": command_examples,
        "commands": commands,
        "examples": examples[:8],
    }
    if ling:
        out["manual_sections"] = ling
        out["linguistic_profile"] = ling
    if parsed.get("manual_sections") and "manual_sections" not in out:
        out["manual_sections"] = parsed["manual_sections"]
    for key in ("persistent_commands", "when_commands", "user_ip", "hashed_ip",
                "num_conversations", "conversations_used", "profile_completeness"):
        if key in parsed:
            out[key] = parsed[key]
    return out


def _parse_profile(raw: str) -> dict[str, Any] | None:
    parsed = parse_json_block(raw)
    if not isinstance(parsed, dict):
        return None
    return normalize_profile(parsed)


def _to_imperative(command: str) -> str:
    """Light rewrite of common descriptive prefixes into second-person imperatives."""

    text = " ".join(str(command or "").split()).strip()
    if not text:
        return ""

    def _fix_consequent(tail: str) -> str:
        tail = re.sub(r"^(the\s+user)\s+", "", tail, flags=re.I).strip(" ,")
        tail = _DESCRIPTIVE_PREFIX_RE.sub("", tail).strip(" ,")
        if not tail:
            return ""
        tail = re.sub(
            r"^(answers|responds|confirms|asks|adds|gives|provides|restates|replies)\b",
            lambda m: (
                m.group(1)[:-1].capitalize()
                if m.group(1).lower().endswith("s") and m.group(1).lower() not in {"answers"}
                else {"answers": "Answer", "responds": "Respond", "replies": "Reply"}.get(
                    m.group(1).lower(), m.group(1).capitalize()
                )
            ),
            tail,
            count=1,
            flags=re.I,
        )
        if tail and tail[0].islower():
            tail = tail[0].upper() + tail[1:]
        if tail and not re.match(
            r"^(Use|Prefer|Omit|Answer|Keep|Write|Ask|Confirm|Restate|Be|Stay|Allow|Add|Drop|Reply|Respond)\b",
            tail,
            re.I,
        ):
            tail = f"Prefer {tail[0].lower() + tail[1:]}"
        return tail.strip()

    # Preserve When-commands' trigger; rewrite the consequent if descriptive.
    if text.lower().startswith("when "):
        parts = text.split(",", 1)
        if len(parts) == 2:
            head, tail = parts[0].strip(), parts[1].strip()
            fixed = _fix_consequent(tail)
            return f"{head}, {fixed}" if fixed else text
        return text
    # Strip leading "The user "
    text = re.sub(r"^(the user)\s+", "", text, flags=re.I)
    m = _DESCRIPTIVE_PREFIX_RE.match(text)
    if not m:
        return text
    rest = text[m.end() :].strip(" ,")
    if not rest:
        return text
    lower = rest.lower()
    replacements = (
        (r"^short[, ]", "Use short "),
        (r"^brief[, ]", "Keep replies brief; "),
        (r"^direct ", "Be direct; use "),
        (r"^lowercase", "Write in lowercase"),
        (r"^polite", "Stay polite"),
        (r"^simple ", "Use simple "),
        (r"^clear[, ]", "Use clear "),
        (r"^typos", "Allow typos"),
        (r"^questions", "Ask questions"),
        (r"^follow-up", "Ask follow-up"),
        (r"^constraints", "Add constraints"),
        (r"^confirmations", "Use short confirmations"),
    )
    for pat, repl in replacements:
        if re.match(pat, lower):
            rest = re.sub(pat, repl, rest, count=1, flags=re.I)
            break
    else:
        rest = _fix_consequent(rest) or rest
    return rest.strip()


def _scrub_example(text: str) -> str:
    """Lightly delexicalize example utterances while keeping style markers."""

    t = " ".join(str(text or "").split()).strip()
    if not t:
        return ""
    t = re.sub(r"\b\d{1,2}/\d{1,2}(/\d{2,4})?\b", "<date>", t)
    t = re.sub(
        r"\b(january|february|march|april|may|june|july|august|september|october|november|december)\b",
        "<date>",
        t,
        flags=re.I,
    )
    t = re.sub(r"\b\d{1,2}:\d{2}\s*(am|pm)?\b", "<time>", t, flags=re.I)
    t = re.sub(r"\b\d+\b", "<n>", t)
    t = re.sub(r"\b[\w.+-]+@[\w.-]+\.\w+\b", "<email>", t)
    t = re.sub(r"https?://\S+", "<url>", t)
    t = _DOMAIN_TOKEN_RE.sub("<place>", t)
    keep = {"i", "ok", "okay", "yes", "no", "hi", "hello", "thanks", "please", "just", "the", "a", "an"}
    parts = t.split(" ")
    scrubbed: list[str] = []
    for i, w in enumerate(parts):
        core = re.sub(r"^[^A-Za-z]+|[^A-Za-z]+$", "", w)
        if (
            i > 0
            and core
            and core[0].isupper()
            and core.lower() not in keep
            and len(core) > 2
        ):
            scrubbed.append(w.replace(core, "<entity>", 1))
        else:
            scrubbed.append(w)
    return " ".join(scrubbed)[:200]


def _sanitize_command_list(commands: list[str], *, when: bool, limit: int) -> list[str]:
    """Drop leaky / empty commands; enforce When-prefix; limit brevity duplicates."""

    out: list[str] = []
    seen: set[str] = set()
    brevity_kept = 0
    for raw in commands:
        cmd = _to_imperative(raw)
        if not cmd:
            continue
        if when and not cmd.lower().startswith("when "):
            cmd = f"When the assistant pushes back, {cmd[0].lower() + cmd[1:]}" if cmd else cmd
        if _DOMAIN_TOKEN_RE.search(cmd):
            continue
        key = cmd.lower()
        if key in seen:
            continue
        if _BREVITY_RE.search(cmd):
            if brevity_kept >= (0 if when else 1):
                continue
            brevity_kept += 1
        seen.add(key)
        out.append(cmd)
        if len(out) >= limit:
            break
    return out


def merge_persistent_when(
    parsed: dict[str, Any],
    *,
    n_persistent: int,
    m_when: int,
) -> dict[str, Any] | None:
    """Normalize dual-bucket extraction into a RealUserSim-compatible profile dict."""

    def _as_str_list(value: Any, limit: int) -> list[str]:
        if isinstance(value, str):
            value = [value]
        if not isinstance(value, (list, tuple)):
            return []
        out: list[str] = []
        for item in value:
            if isinstance(item, dict):
                item = item.get("command") or item.get("text") or item.get("rule") or ""
            text = str(item or "").strip()
            if text:
                out.append(text)
            if len(out) >= limit:
                break
        return out

    persistent = _as_str_list(parsed.get("persistent_commands"), max(0, n_persistent) + 4)
    when = _as_str_list(parsed.get("when_commands"), max(0, m_when) + 4)
    # Fallback if the model returned a flat commands list only.
    if not persistent and not when:
        flat = _as_str_list(parsed.get("commands"), max(0, n_persistent + m_when) + 4)
        persistent, when = flat[: n_persistent + 4], flat[n_persistent:]
    persistent = _sanitize_command_list(persistent, when=False, limit=max(0, n_persistent))
    when = _sanitize_command_list(when, when=True, limit=max(0, m_when))
    commands = persistent + when
    examples = [_scrub_example(e) for e in _as_str_list(parsed.get("examples"), 8)]
    examples = [e for e in examples if e][:4]
    if not commands and not examples:
        return None
    demo = parsed.get("demographics") if isinstance(parsed.get("demographics"), dict) else {}
    # Scrub domain tokens from demographics strings (CUE Stage-0 / refinement path).
    clean_demo: dict[str, Any] = {}
    for k in ("expertise", "tone", "verbosity", "background"):
        val = _DOMAIN_TOKEN_RE.sub("general", str(demo.get(k) or "").strip())
        clean_demo[k] = val
    for k, v in demo.items():
        if k not in clean_demo:
            clean_demo[k] = v
    command_examples = [{"command": c, "examples": []} for c in commands]
    if examples and command_examples:
        command_examples[0]["examples"] = examples[:4]
    return {
        "demographics": clean_demo,
        "commands": commands,
        "persistent_commands": persistent,
        "when_commands": when,
        "examples": examples,
        "command_examples": command_examples,
    }


# Back-compat alias for callers that still use the private name.
_merge_persistent_when = merge_persistent_when


def extract_profile_from_user_text(
    user_text: str,
    *,
    model: str = DEFAULT_PROFILER_MODEL,
    api_key_env: str | None = "OPENAI_API_KEY",
    api_base: str | None = None,
    temperature: float = 0.3,
    max_tokens: int = 1600,
    include_examples: bool = True,
) -> dict[str, Any] | None:
    """Run the RealUserSim Executable Persona Manual extraction over a user-turn transcript."""

    if not (user_text or "").strip():
        return None
    system = _EXTRACTION_PROMPT
    if not include_examples:
        system = (
            _EXTRACTION_PROMPT
            + "\n- CRITICAL: set every command_examples[].examples to []. "
            "Do NOT quote or copy any user message text into examples "
            "(paired-eval leakage control).\n"
        )
    try:
        raw = chat(
            [
                {"role": "system", "content": system},
                {"role": "user", "content": f"Conversation user turns:\n{user_text}"},
            ],
            model=model or DEFAULT_PROFILER_MODEL,
            api_key_env=api_key_env,
            api_base=api_base,
            temperature=temperature,
            max_tokens=max_tokens,
        )
    except Exception as exc:  # noqa: BLE001 - skip flaky extractions
        warn("realusersim", f"extraction failed: {exc}")
        return None
    parsed = _parse_profile(raw)
    if parsed is not None and not include_examples:
        from cue_training.baselines.realusersim.format_persona import strip_profile_examples

        parsed = strip_profile_examples(parsed)
    return parsed


async def aextract_profile_from_user_text(
    user_text: str,
    *,
    model: str = DEFAULT_PROFILER_MODEL,
    api_key_env: str | None = "OPENAI_API_KEY",
    api_base: str | None = None,
    temperature: float = 0.3,
    max_tokens: int = 1600,
) -> dict[str, Any] | None:
    """Async counterpart of :func:`extract_profile_from_user_text` via ``litellm.acompletion``."""

    if not (user_text or "").strip():
        return None
    try:
        raw = await achat(
            [
                {"role": "system", "content": _EXTRACTION_PROMPT},
                {"role": "user", "content": f"Conversation user turns:\n{user_text}"},
            ],
            model=model or DEFAULT_PROFILER_MODEL,
            api_key_env=api_key_env,
            api_base=api_base,
            temperature=temperature,
            max_tokens=max_tokens,
        )
    except Exception as exc:  # noqa: BLE001 - skip flaky extractions
        warn("realusersim", f"extraction failed: {exc}")
        return None
    return _parse_profile(raw)


async def aextract_profile_persistent_and_when(
    user_text: str,
    *,
    n_persistent: int = 10,
    m_when: int = 5,
    model: str,
    api_key_env: str | None = "OPENAI_API_KEY",
    api_base: str | None = None,
    temperature: float = 0.3,
    max_tokens: int = 800,
    timeout: float | None = 120.0,
    num_retries: int = 1,
    contrastive: bool = False,
) -> dict[str, Any] | None:
    """Async extraction with N persistent style commands + M conditional when-commands."""

    if not (user_text or "").strip():
        return None
    try:
        raw = await achat(
            [
                {
                    "role": "system",
                    "content": _persistent_when_prompt(
                        n_persistent=max(0, n_persistent), m_when=max(0, m_when)
                    )
                    + (
                        "\nThe evidence contains TARGET, POSITIVE, and NEGATIVE groups. "
                        "Commands should explain why TARGET is closer to POSITIVE than NEGATIVE. "
                        "Never use TARGET turns as output examples."
                        if contrastive else ""
                    ),
                },
                {"role": "user", "content": user_text},
            ],
            model=model,
            api_key_env=api_key_env,
            api_base=api_base,
            temperature=temperature,
            max_tokens=max_tokens,
            timeout=timeout,
            num_retries=num_retries,
        )
    except Exception as exc:  # noqa: BLE001
        warn("realusersim", f"persistent/when extraction failed: {exc}")
        return None
    parsed = parse_json_block(raw)
    if not isinstance(parsed, dict):
        return None
    return _merge_persistent_when(parsed, n_persistent=n_persistent, m_when=m_when)


async def arerank_profile_persistent_and_when(
    evidence: str,
    *,
    persistent_candidates: list[str],
    when_candidates: list[str],
    n_persistent: int,
    m_when: int,
    model: str,
    api_key_env: str | None = "OPENAI_API_KEY",
    api_base: str | None = None,
    timeout: float | None = 120.0,
    num_retries: int = 1,
) -> dict[str, list[str]] | None:
    """LLM-select exact candidate commands with contrastive target/positive/negative evidence."""

    if not (persistent_candidates or when_candidates):
        return None
    candidate_text = (
        f"{evidence}\n\nPERSISTENT CANDIDATES:\n"
        + "\n".join(f"- {command}" for command in persistent_candidates)
        + "\n\nWHEN CANDIDATES:\n"
        + "\n".join(f"- {command}" for command in when_candidates)
    )
    try:
        raw = await achat(
            [
                {
                    "role": "system",
                    "content": contrastive_rerank_prompt(
                        n_persistent=max(0, n_persistent), m_when=max(0, m_when)
                    ),
                },
                {"role": "user", "content": candidate_text},
            ],
            model=model,
            api_key_env=api_key_env,
            api_base=api_base,
            temperature=0.0,
            max_tokens=600,
            timeout=timeout,
            num_retries=num_retries,
        )
    except Exception as exc:  # noqa: BLE001
        warn("realusersim", f"contrastive command rerank failed: {exc}")
        return None
    parsed = parse_json_block(raw)
    if not isinstance(parsed, dict):
        return None
    # Keep only exact candidates, preserving candidate order and avoiding prompt-injected rewrites.
    def _select(candidates: list[str], requested: Any, limit: int) -> list[str]:
        wanted = {str(value).strip() for value in (requested or []) if str(value).strip()}
        return [candidate for candidate in candidates if candidate in wanted][:limit]
    return {
        "persistent_commands": _select(
            persistent_candidates, parsed.get("persistent_commands"), n_persistent
        ),
        "when_commands": _select(when_candidates, parsed.get("when_commands"), m_when),
    }


def extract_profiles(cfg: TrainConfig) -> Path:
    """Default: import the authors' published Salesforce/RealUserSim profiles.

    Knobs via ``cfg.extra`` / env:
      - ``rebuild_from_wildchat`` / ``REALUSERSIM_REBUILD=1``: run the full
        WildChat-4.8M curate → per-user GPT-4o extraction instead of importing.
      - ``use_published_profiles=False`` also forces a rebuild.
      - ``skip_curate`` if ``wildchat_filtered_trajs.jsonl`` already exists.
      - ``limit``: cap users (None = all; legacy path used 200).
    """

    out_path = cfg.output_dir / PROFILES_FILE
    extra = dict(cfg.extra or {})
    rebuild = bool(
        extra.get("rebuild_from_wildchat")
        or extra.get("legacy_per_conversation")
        or extra.get("use_published_profiles") is False
        or str(os.environ.get("REALUSERSIM_REBUILD", "")).lower() in {"1", "true", "yes"}
    )
    if not rebuild:
        return import_published_profiles(cfg.output_dir)

    # Explicit legacy path: extract one profile per training conversation from data_path.
    if extra.get("legacy_per_conversation") and cfg.data_path is not None:
        return _extract_profiles_legacy(cfg, out_path)

    return build_paper_profiles(cfg, out_path)


def import_published_profiles(output_dir: Path) -> Path:
    """Copy the authors' released profiles verbatim (Salesforce/RealUserSim).

    Rows are written exactly as published; ``manual_sections`` mirrors the
    ``linguistic_profile`` field only so the existing formatter/simulator read it
    unchanged (the source key is preserved too).
    """

    download = globals().get("hf_hub_download")
    if download is None:
        from huggingface_hub import hf_hub_download as download

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / PROFILES_FILE
    src = download(
        repo_id="Salesforce/RealUserSim",
        filename="profiles/consolidated_user_profiles.jsonl",
        repo_type="dataset",
    )
    profiles: list[dict[str, Any]] = []
    with open(src, encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            raw = json.loads(line)
            if not isinstance(raw, dict):
                continue
            profile = dict(raw)  # keep every published field verbatim
            ling = str(raw.get("linguistic_profile") or "").strip()
            if ling and not profile.get("manual_sections"):
                profile["manual_sections"] = ling
            profiles.append(profile)
    write_jsonl(out_path, profiles)
    log("realusersim", f"imported {len(profiles)} published profiles verbatim -> {out_path}")
    return out_path


def build_paper_profiles(cfg: TrainConfig, out_path: Path | None = None) -> Path:
    """Curate WildChat-4.8M (App. A.1 stages 1–2) then extract one profile per user."""

    from cue_training.baselines.realusersim.wildchat_curate import (
        FILTERED_TRAJS_FILE,
        curate_wildchat,
        group_user_transcripts,
    )

    artifacts = Path(cfg.output_dir)
    artifacts.mkdir(parents=True, exist_ok=True)
    out_path = Path(out_path or (artifacts / PROFILES_FILE))
    trajs = artifacts / FILTERED_TRAJS_FILE
    extra = dict(cfg.extra or {})
    skip_curate = bool(extra.get("skip_curate")) and trajs.is_file()
    if not skip_curate:
        curate_wildchat(
            artifacts,
            repo=str(extra.get("wildchat_repo") or "allenai/WildChat-4.8M"),
            streaming=bool(extra.get("streaming", True)),
            max_rows=extra.get("max_rows"),
        )
    elif not trajs.is_file():
        raise FileNotFoundError(f"missing curated trajs at {trajs}; re-run without skip_curate")

    by_user = group_user_transcripts(trajs)
    user_ids = sorted(by_user.keys())
    limit = cfg.limit
    if limit is not None:
        user_ids = user_ids[: int(limit)]
    log(
        "realusersim",
        f"extracting personas for {len(user_ids):,} / {len(by_user):,} users "
        f"(profiler={cfg.sim_model or DEFAULT_PROFILER_MODEL})",
    )

    def _extract(uid: str) -> dict[str, Any] | None:
        profile = extract_profile_from_user_text(
            by_user[uid],
            model=cfg.sim_model or DEFAULT_PROFILER_MODEL,
            api_key_env=cfg.api_key_env,
            api_base=cfg.api_base,
        )
        if profile is None:
            return None
        profile = dict(profile)
        profile["hashed_ip"] = uid
        return profile

    workers = max(1, int(cfg.batch_size or 1))
    if workers > 1:
        with ThreadPoolExecutor(max_workers=workers) as executor:
            results = list(executor.map(_extract, user_ids))
    else:
        results = [_extract(uid) for uid in user_ids]
    profiles = [p for p in results if p is not None]
    write_jsonl(out_path, profiles)
    log("realusersim", f"wrote {len(profiles)} persona profiles -> {out_path}")
    return out_path


def _extract_profiles_legacy(cfg: TrainConfig, out_path: Path) -> Path:
    """One profile per training conversation (pre-paper-pool path; opt-in only)."""

    limit = cfg.limit if cfg.limit is not None else 200
    user_texts = [
        text
        for conversation in training_conversations(
            cfg.data_path, limit=limit, jsonl_names=tuple(cfg.jsonl_names)
        )
        if (text := _user_only(conversation)).strip()
    ]

    def _extract(user_text: str) -> dict | None:
        return extract_profile_from_user_text(
            user_text,
            model=cfg.sim_model or DEFAULT_PROFILER_MODEL,
            api_key_env=cfg.api_key_env,
            api_base=cfg.api_base,
        )

    workers = max(1, cfg.batch_size)
    if workers > 1:
        with ThreadPoolExecutor(max_workers=workers) as executor:
            results = list(executor.map(_extract, user_texts))
    else:
        results = [_extract(text) for text in user_texts]
    profiles = [profile for profile in results if profile is not None]
    write_jsonl(out_path, profiles)
    log("realusersim", f"wrote {len(profiles)} legacy persona profiles -> {out_path}")
    return out_path


def load_profiles(path: Path) -> list[dict]:
    return read_jsonl(path)
