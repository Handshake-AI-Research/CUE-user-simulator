"""LLM propose / discover / classify for FMA."""

from __future__ import annotations

import json
from collections.abc import Callable, Iterable
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, TypeVar

from cue_training.baselines.common.llm import (
    accumulate_chat_cost,
    chat,
    parse_json_block,
)
from cue_training.evaluation.common.io import read_jsonl, write_jsonl
from cue_training.evaluation.fma.conversation import format_transcript
from cue_training.evaluation.fma.paths import (
    api_key_env_for_model,
    batch_dir,
    default_model,
)
from cue_training.evaluation.fma.prompts_util import render
from cue_training.evaluation.fma.sample import load_sample
from cue_training.evaluation.fma.store import load_taxonomy, save_taxonomy, upsert_mode

ChatFn = Callable[..., str]
_T = TypeVar("_T")
_R = TypeVar("_R")

DEFAULT_WORKERS = 16


def _map_concurrent[T, R](
    fn: Callable[[_T], _R],
    items: list[_T],
    *,
    workers: int = DEFAULT_WORKERS,
    desc: str = "fma",
    on_result: Callable[[_R], None] | None = None,
) -> list[_R]:
    """Run ``fn`` over ``items`` with a thread pool; preserve input order.

    ``on_result`` is invoked on the calling thread as each result lands, so callers
    can checkpoint partial progress before an interrupt discards the whole pass.
    """

    if not items:
        return []
    workers = max(1, min(int(workers), len(items)))
    if workers == 1:
        try:
            from tqdm import tqdm
        except ImportError:  # pragma: no cover
            tqdm = None  # type: ignore[assignment,misc]
        rows: Iterable[_T] = items
        if tqdm is not None:
            rows = tqdm(items, desc=desc, unit="traj")
        seq: list[_R] = []
        for item in rows:
            res = fn(item)
            if on_result is not None:
                on_result(res)
            seq.append(res)
        return seq

    try:
        from tqdm import tqdm
    except ImportError:  # pragma: no cover
        tqdm = None  # type: ignore[assignment,misc]

    out: list[_R | None] = [None] * len(items)
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futs = {pool.submit(fn, item): i for i, item in enumerate(items)}
        done = as_completed(futs)
        if tqdm is not None:
            done = tqdm(done, total=len(futs), desc=desc, unit="traj")
        for fut in done:
            res = fut.result()
            if on_result is not None:
                on_result(res)
            out[futs[fut]] = res
    return out  # type: ignore[return-value]


def _examples_from_taxonomy(
    taxonomy: list[dict[str, Any]], *, k: int = 2
) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for mode in taxonomy:
        for ex in (mode.get("examples") or [])[:k]:
            out.append(
                {
                    "label": mode.get("name"),
                    "explanation": ex.get("explanation") or "",
                    "turn_indices": ex.get("turn_indices") or [],
                    "snippet": ex.get("snippet") or ex.get("excerpt") or "",
                }
            )
    return out[: 3 * k]


def _reward_report_text(report: Any) -> str:
    if report is None:
        return ""
    return json.dumps(report, indent=2, ensure_ascii=False)


def propose_one(
    row: dict[str, Any],
    *,
    taxonomy: list[dict[str, Any]],
    model: str,
    chat_fn: ChatFn = chat,
    max_tokens: int = 8192,
) -> dict[str, Any]:
    prompt = render(
        "label_failure.jinja",
        examples=_examples_from_taxonomy(taxonomy),
        taxonomy=taxonomy,
        source_id=row.get("source_id"),
        domain=row.get("domain"),
        task_id=row.get("task_id"),
        task_description=row.get("task_description") or "",
        success_criteria=row.get("success_criteria") or "",
        reward_report=_reward_report_text(row.get("reward_report")),
        transcript=format_transcript(row.get("conversation") or [], max_chars=24_000),
    )
    raw = chat_fn(
        [{"role": "user", "content": prompt}],
        model=model,
        temperature=0.2,
        max_tokens=max_tokens,
        api_key_env=api_key_env_for_model(model),
    )
    parsed = parse_json_block(raw) or {}
    label = str(parsed.get("label") or "Uncategorized").strip() or "Uncategorized"
    turn_indices = parsed.get("turn_indices") or []
    if not isinstance(turn_indices, list):
        turn_indices = []
    turn_indices = [int(x) for x in turn_indices if str(x).lstrip("-").isdigit()]
    return {
        "primary_key": row["primary_key"],
        "episode_id": row.get("episode_id"),
        "source_id": row.get("source_id"),
        "label": label,
        "explanation": str(parsed.get("explanation") or "").strip(),
        "turn_indices": turn_indices,
        "excerpt": str(parsed.get("excerpt") or "").strip(),
        "raw": raw,
    }


def propose_batch(
    run_dir: Path,
    *,
    batch: int,
    model: str | None = None,
    chat_fn: ChatFn = chat,
    workers: int = DEFAULT_WORKERS,
) -> dict[str, Any]:
    taxonomy = load_taxonomy(run_dir)
    sample = load_sample(run_dir, batch)
    model_name = model or default_model()
    with accumulate_chat_cost() as cost:
        proposals = _map_concurrent(
            lambda row: propose_one(
                row, taxonomy=taxonomy, model=model_name, chat_fn=chat_fn
            ),
            sample,
            workers=workers,
            desc=f"cue-fma propose batch={batch}",
        )
    print(cost.format_line(label="cue-fma propose"), flush=True)
    out = batch_dir(run_dir, batch) / "proposals.jsonl"
    write_jsonl(str(out), proposals)
    # Seed taxonomy with open labels (descriptions filled later by discover / human).
    for prop in proposals:
        upsert_mode(taxonomy, name=prop["label"], description="")
    save_taxonomy(run_dir, taxonomy)
    return {
        "batch": batch,
        "n_proposals": len(proposals),
        "workers": max(1, min(int(workers), max(1, len(sample)))),
        "path": str(out),
        "cost": cost.summary(),
    }


def discover_classes(
    run_dir: Path,
    *,
    batch: int | None = None,
    model: str | None = None,
    chat_fn: ChatFn = chat,
) -> dict[str, Any]:
    taxonomy = load_taxonomy(run_dir)
    items: list[dict[str, Any]] = []
    if batch is not None:
        path = batch_dir(run_dir, batch) / "proposals.jsonl"
        items = [
            {"label": r.get("label"), "explanation": r.get("explanation")}
            for r in read_jsonl(str(path))
        ]
    else:
        for mode in taxonomy:
            items.append(
                {"label": mode.get("name"), "explanation": mode.get("description")}
            )
    prompt = render("discover_classes.jinja", taxonomy=taxonomy, items=items)
    raw = chat_fn(
        [{"role": "user", "content": prompt}],
        model=model or default_model(),
        temperature=0.2,
        max_tokens=8192,
        api_key_env=api_key_env_for_model(model or default_model()),
    )
    parsed = parse_json_block(raw) or {}
    categories = parsed.get("categories") or []
    for cat in categories:
        name = str(cat.get("name") or "").strip()
        if not name:
            continue
        upsert_mode(
            taxonomy,
            name=name,
            description=str(cat.get("description") or "").strip(),
        )
    save_taxonomy(run_dir, taxonomy)
    return {"n_categories": len(categories), "raw": raw}


def classify_one(
    row: dict[str, Any],
    *,
    taxonomy: list[dict[str, Any]],
    model: str,
    chat_fn: ChatFn = chat,
    reasoning_effort: str | None = None,
) -> dict[str, Any]:
    prompt = render(
        "classify.jinja",
        taxonomy=taxonomy,
        source_id=row.get("source_id"),
        domain=row.get("domain"),
        task_id=row.get("task_id"),
        task_description=row.get("task_description") or "",
        success_criteria=row.get("success_criteria") or "",
        reward_report=_reward_report_text(row.get("reward_report")),
        transcript=format_transcript(row.get("conversation") or [], max_chars=24_000),
    )
    # Picking one frozen taxonomy label does not need deep reasoning; leaving a
    # reasoning model at its default effort burns minutes per long transcript.
    extra = {"reasoning_effort": reasoning_effort} if reasoning_effort else {}
    raw = chat_fn(
        [{"role": "user", "content": prompt}],
        model=model,
        temperature=0.1,
        max_tokens=8192,
        api_key_env=api_key_env_for_model(model),
        **extra,
    )
    parsed = parse_json_block(raw) or {}
    names = {m["name"] for m in taxonomy}
    label = str(parsed.get("label") or "Uncategorized").strip()
    if label not in names:
        label = "Uncategorized"
    turn_indices = parsed.get("turn_indices") or []
    if not isinstance(turn_indices, list):
        turn_indices = []
    turn_indices = [int(x) for x in turn_indices if str(x).lstrip("-").isdigit()]
    return {
        "primary_key": row["primary_key"],
        "episode_id": row.get("episode_id"),
        "source_id": row.get("source_id"),
        "label": label,
        "explanation": str(parsed.get("explanation") or "").strip(),
        "turn_indices": turn_indices,
        "excerpt": str(parsed.get("excerpt") or "").strip(),
        "origin": "tag",
    }
