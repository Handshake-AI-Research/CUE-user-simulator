"""RealUserSim Appendix A.1 WildChat curation (stages 1–2).

Paper (arXiv:2605.20204): start from WildChat-4.8M (3.2M non-toxic), filter to
21,637 multi-turn English GPT-4o trajectories across 7,311 unique users, then
build one Executable Persona Manual per user (Appendix A.3).

Stages 3–4 (domain/task tagging + final 13,998 curation) are for the PT3 eval
subsets only; profile construction uses the stage-2 pool.
"""

from __future__ import annotations

import json
import re
from collections import defaultdict
from collections.abc import Iterable, Iterator, Mapping
from pathlib import Path
from typing import Any

from cue_training.runlog.log import log, warn

WILDCHAT_REPO = "allenai/WildChat-4.8M"
FILTERED_TRAJS_FILE = "wildchat_filtered_trajs.jsonl"
FILTER_STATS_FILE = "wildchat_filter_stats.json"

# Paper targets (Appendix A.1 stage 2). Exact replay depends on HF revision / language
# detector; we record actual counts and warn on large deviations.
PAPER_N_TRAJS = 21_637
PAPER_N_USERS = 7_311

_GPT4O_RE = re.compile(r"gpt-4o(?![-_]?mini)", re.I)
_GREETING_THANKS_RE = re.compile(
    r"^\s*("
    r"hi|hello|hey|hiya|yo|sup|"
    r"thanks|thank\s*you|thx|ty|tysm|"
    r"bye|goodbye|good\s*bye|see\s*ya|"
    r"ok\s*thanks|okay\s*thanks|ok\s*thank\s*you"
    r")[\s!.?]*$",
    re.I,
)
_WORD_RE = re.compile(r"\S+")


def _word_count(text: str) -> int:
    return len(_WORD_RE.findall(text or ""))


def _is_greeting_or_thanks(text: str) -> bool:
    t = " ".join((text or "").split()).strip()
    if not t or _word_count(t) > 5:
        return False
    return bool(_GREETING_THANKS_RE.match(t))


def trim_greeting_thanks(messages: list[Mapping[str, Any]]) -> list[dict[str, str]]:
    """Drop bookend greeting/thank-you rounds of ≤5 words (paper stage 2)."""

    turns: list[dict[str, str]] = []
    for msg in messages:
        if not isinstance(msg, Mapping):
            continue
        role = str(msg.get("role") or "").strip().lower()
        if role not in {"user", "assistant", "human", "gpt", "system"}:
            continue
        if role in {"human"}:
            role = "user"
        if role in {"gpt"}:
            role = "assistant"
        content = str(msg.get("content") or "").strip()
        if not content:
            continue
        turns.append({"role": role, "content": content})

    while turns and _is_greeting_or_thanks(turns[0]["content"]):
        turns.pop(0)
    while turns and _is_greeting_or_thanks(turns[-1]["content"]):
        turns.pop()
    return turns


def conversation_is_english(messages: list[Mapping[str, Any]], *, min_conf: float = 0.7) -> bool:
    """English gate approximating paper fastText ≥0.7 via WildChat turn ``language`` labels.

    Prefer user-turn languages; fall back to all turns. Optional ``language_confidence`` /
    ``lang_score`` fields are honored when present.
    """

    labels: list[str] = []
    for msg in messages:
        if not isinstance(msg, Mapping):
            continue
        role = str(msg.get("role") or "").strip().lower()
        lang = str(msg.get("language") or "").strip().lower()
        if not lang:
            continue
        conf_raw = msg.get("language_confidence", msg.get("lang_score", msg.get("confidence")))
        if conf_raw is not None:
            try:
                if float(conf_raw) < float(min_conf):
                    continue
            except (TypeError, ValueError):
                pass
        if role in {"user", "human"} or not labels:
            labels.append(lang)
    if not labels:
        # No per-turn language: accept (WildChat-4.8M usually has labels; avoid false drops).
        return True
    english = sum(1 for x in labels if x in {"en", "eng", "english"})
    return english >= max(1, (len(labels) + 1) // 2)


def is_gpt4o_model(model: Any) -> bool:
    return bool(_GPT4O_RE.search(str(model or "")))


def passes_stage1(row: Mapping[str, Any]) -> bool:
    """Multi-turn (≥2 messages) English."""

    conv = row.get("conversation")
    if not isinstance(conv, list) or len(conv) < 2:
        return False
    return conversation_is_english(conv)


def passes_stage2(row: Mapping[str, Any]) -> tuple[bool, list[dict[str, str]]]:
    """GPT-4o + ≥3 substantive turns after greeting/thanks trim."""

    if not is_gpt4o_model(row.get("model")):
        return False, []
    conv = row.get("conversation")
    if not isinstance(conv, list):
        return False, []
    trimmed = trim_greeting_thanks(conv)
    if len(trimmed) < 3:
        return False, []
    return True, trimmed


def user_key(row: Mapping[str, Any]) -> str:
    """Stable user id: conversation-level hashed_ip (paper unique-user key)."""

    ip = row.get("hashed_ip")
    if ip:
        return str(ip)
    conv = row.get("conversation")
    if isinstance(conv, list):
        for msg in conv:
            if isinstance(msg, Mapping) and msg.get("hashed_ip"):
                return str(msg["hashed_ip"])
    return str(row.get("conversation_hash") or row.get("conversation_id") or "unknown")


def filter_row(row: Mapping[str, Any]) -> dict[str, Any] | None:
    """Apply paper stages 1–2; return a normalized trajectory record or None."""

    if not isinstance(row, Mapping):
        return None
    if bool(row.get("toxic")):
        return None
    if not passes_stage1(row):
        return None
    ok, trimmed = passes_stage2(row)
    if not ok:
        return None
    return {
        "conversation_hash": row.get("conversation_hash"),
        "model": row.get("model"),
        "hashed_ip": user_key(row),
        "turns": trimmed,
        "n_turns": len(trimmed),
        "n_user_turns": sum(1 for t in trimmed if t["role"] == "user"),
    }


def iter_wildchat_rows(
    *,
    repo: str = WILDCHAT_REPO,
    split: str = "train",
    streaming: bool = True,
    max_rows: int | None = None,
) -> Iterator[Mapping[str, Any]]:
    from datasets import load_dataset

    ds = load_dataset(repo, split=split, streaming=streaming)
    for i, row in enumerate(ds):
        if max_rows is not None and i >= max_rows:
            break
        yield row


def curate_wildchat(
    output_dir: Path,
    *,
    repo: str = WILDCHAT_REPO,
    streaming: bool = True,
    max_rows: int | None = None,
    rows: Iterable[Mapping[str, Any]] | None = None,
) -> tuple[Path, dict[str, Any]]:
    """Stream WildChat-4.8M, write stage-2 trajectories + stats under ``output_dir``."""

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / FILTERED_TRAJS_FILE
    scanned = 0
    kept = 0
    users: set[str] = set()
    source = rows if rows is not None else iter_wildchat_rows(
        repo=repo, streaming=streaming, max_rows=max_rows
    )

    with out_path.open("w", encoding="utf-8") as handle:
        for row in source:
            scanned += 1
            if scanned % 100_000 == 0:
                log(
                    "realusersim.curate",
                    f"scanned={scanned:,} kept={kept:,} users={len(users):,}",
                )
            rec = filter_row(row)
            if rec is None:
                continue
            users.add(str(rec["hashed_ip"]))
            handle.write(json.dumps(rec, ensure_ascii=False) + "\n")
            kept += 1

    stats = {
        "repo": repo,
        "scanned": scanned,
        "kept_trajs": kept,
        "unique_users": len(users),
        "paper_trajs": PAPER_N_TRAJS,
        "paper_users": PAPER_N_USERS,
        "output": str(out_path),
    }
    (output_dir / FILTER_STATS_FILE).write_text(
        json.dumps(stats, indent=2) + "\n", encoding="utf-8"
    )
    log(
        "realusersim.curate",
        f"kept {kept:,} trajs / {len(users):,} users "
        f"(paper targets {PAPER_N_TRAJS:,} / {PAPER_N_USERS:,}) -> {out_path}",
    )
    if kept and (
        abs(kept - PAPER_N_TRAJS) / PAPER_N_TRAJS > 0.25
        or abs(len(users) - PAPER_N_USERS) / PAPER_N_USERS > 0.25
    ):
        warn(
            "realusersim.curate",
            "filter counts deviate >25% from paper Appendix A.1; check model/language gates.",
        )
    return out_path, stats


def group_user_transcripts(trajs_path: Path) -> dict[str, str]:
    """Concatenate user turns across a user's filtered trajectories (paper A.3 input)."""

    by_user: dict[str, list[str]] = defaultdict(list)
    with Path(trajs_path).open(encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            uid = str(rec.get("hashed_ip") or "")
            if not uid:
                continue
            for turn in rec.get("turns") or []:
                if turn.get("role") == "user" and str(turn.get("content") or "").strip():
                    by_user[uid].append(str(turn["content"]).strip())
    return {
        uid: "\n".join(f"- {t}" for t in turns)
        for uid, turns in by_user.items()
        if turns
    }
