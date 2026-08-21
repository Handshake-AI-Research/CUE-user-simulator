"""Add style-kind commands to existing data-annotation manuals without regenerating them."""

from __future__ import annotations

import argparse
import json
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

from tqdm import tqdm

from cue_training.data.schema import validate_manual
from cue_training.data.streaming import discover_jsonl_files, iter_jsonl, write_jsonl_atomic
from cue_training.preprocessing.llm import complete_json
from cue_training.preprocessing.prompts import format_turns
from cue_training.utils.log import log, warn

TAG = "style-augment"

STYLE_SYSTEM = """You extract SURFACE WRITING STYLE rules for a user simulator.
Return STRICT JSON:
{
  "commands": [
    {
      "text": "second-person imperative style rule",
      "kind": "style",
      "form": "general",
      "examples": ["1-3 short stylistic exemplars with placeholders if needed"],
      "evidence_turn_ids": ["turn_id from TARGET user turns"]
    }
  ]
}
Hard rules:
- Emit exactly the requested number of style commands.
- Style ONLY: casing, punctuation, typos/misspellings, fragments vs full sentences,
  verbosity, humor, gratitude density, filler words, emoji/exclamation habits.
- Do NOT write dialogue-policy rules (missing details, closing, out-of-order replies,
  booking flow, pushback). Those belong elsewhere.
- Do NOT mention domains/slots (flights, hotels, bookings, prices, IDs).
- Prefer second-person imperatives: "Use lowercase fragments…", "Omit end punctuation…".
- Every command needs evidence_turn_ids from the provided TARGET user turns.
"""


def build_style_messages(
    *,
    turns: list[dict[str, Any]],
    n_style: int,
) -> list[dict[str, str]]:
    user_ids = [
        str(t.get("turn_id"))
        for t in turns
        if t.get("role") == "user" and t.get("turn_id")
    ]
    user = (
        f"Extract {n_style} style commands from this TARGET session.\n"
        f"user_turn_ids={user_ids}\n"
        f"{format_turns(turns, max_chars=350)}\n\n"
        "Return only the JSON object."
    )
    return [
        {"role": "system", "content": STYLE_SYSTEM},
        {"role": "user", "content": user},
    ]


def _has_style_commands(manual: dict[str, Any]) -> bool:
    commands = manual.get("commands") or []
    return any(str(c.get("kind") or "") == "style" for c in commands if isinstance(c, dict))


def extract_style_commands(
    *,
    turns: list[dict[str, Any]],
    model: str,
    n_style: int = 4,
    api_base: str | None = None,
    temperature: float = 0.2,
) -> list[dict[str, Any]]:
    parsed = complete_json(
        model=model,
        messages=build_style_messages(turns=turns, n_style=n_style),
        temperature=temperature,
        max_tokens=700,
        api_base=api_base,
    )
    raw = parsed.get("commands") if isinstance(parsed, dict) else None
    if not isinstance(raw, list):
        raise ValueError("style extraction missing commands list")
    out: list[dict[str, Any]] = []
    for entry in raw:
        if isinstance(entry, str):
            entry = {"text": entry}
        if not isinstance(entry, dict):
            continue
        text = str(entry.get("text") or "").strip()
        if not text:
            continue
        out.append(
            {
                "text": text,
                "kind": "style",
                "form": "general",
                "examples": [
                    str(x).strip()
                    for x in (entry.get("examples") or [])
                    if str(x).strip()
                ][:3],
                "evidence_turn_ids": [
                    str(x) for x in (entry.get("evidence_turn_ids") or []) if str(x).strip()
                ],
            }
        )
        if len(out) >= n_style:
            break
    if not out:
        raise ValueError("style extraction returned no usable commands")
    return out


def augment_record(
    record: dict[str, Any],
    *,
    model: str,
    n_style: int,
    api_base: str | None,
    force: bool,
) -> dict[str, Any]:
    manual = record.get("persona_manual")
    if not isinstance(manual, dict):
        return record
    if _has_style_commands(manual) and not force:
        return record
    style_cmds = extract_style_commands(
        turns=list(record.get("turns") or []),
        model=model,
        n_style=n_style,
        api_base=api_base,
    )
    kept = [
        cmd
        for cmd in (manual.get("commands") or [])
        if not (isinstance(cmd, dict) and str(cmd.get("kind") or "") == "style")
    ]
    updated = {**manual, "commands": kept + style_cmds}
    record = {**record, "persona_manual": validate_manual(updated)}
    return record


def augment_file(
    src: Path,
    dst: Path,
    *,
    model: str,
    n_style: int,
    api_base: str | None,
    force: bool,
    workers: int,
) -> tuple[int, int]:
    records = list(iter_jsonl(src, validate=False))
    out: list[dict[str, Any] | None] = [None] * len(records)
    updated = 0

    def _one(index: int, record: dict[str, Any]) -> tuple[int, dict[str, Any], bool]:
        before = json.dumps(record.get("persona_manual") or {}, sort_keys=True)
        try:
            new_rec = augment_record(
                record,
                model=model,
                n_style=n_style,
                api_base=api_base,
                force=force,
            )
        except Exception as exc:  # noqa: BLE001
            warn(TAG, f"{src.name} id={record.get('id')}: {exc}")
            return index, record, False
        after = json.dumps(new_rec.get("persona_manual") or {}, sort_keys=True)
        return index, new_rec, after != before

    with ThreadPoolExecutor(max_workers=max(1, workers)) as pool:
        futures = [pool.submit(_one, i, rec) for i, rec in enumerate(records)]
        for fut in tqdm(as_completed(futures), total=len(futures), desc=src.name):
            index, record, changed = fut.result()
            out[index] = record
            updated += int(changed)
    write_jsonl_atomic(dst, [r for r in out if r is not None])
    return len(records), updated


def run_style_augment(
    *,
    data_root: str | Path,
    output_root: str | Path,
    splits: list[str],
    model: str,
    n_style: int = 4,
    api_base: str | None = None,
    force: bool = False,
    workers: int = 8,
) -> Path:
    data_root = Path(data_root)
    output_root = Path(output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    total = changed = 0
    for split in splits:
        files = discover_jsonl_files(data_root, split=split)
        if not files:
            warn(TAG, f"no {split}.jsonl under {data_root}")
            continue
        for src in files:
            rel = src.relative_to(data_root)
            dst = output_root / rel
            dst.parent.mkdir(parents=True, exist_ok=True)
            n, u = augment_file(
                src,
                dst,
                model=model,
                n_style=n_style,
                api_base=api_base,
                force=force,
                workers=workers,
            )
            total += n
            changed += u
            log(TAG, f"{rel}: wrote {n} records ({u} style-augmented) -> {dst}")
    log(TAG, f"done records={total} augmented={changed} -> {output_root}")
    return output_root


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data_root", required=True)
    p.add_argument("--output_root", required=True)
    p.add_argument("--splits", default="train,validation")
    p.add_argument("--model", default="gpt-5.4-mini")
    p.add_argument("--n_style", type=int, default=4)
    p.add_argument("--api_base", default=None)
    p.add_argument("--force", action="store_true")
    p.add_argument("--workers", type=int, default=8)
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    run_style_augment(
        data_root=args.data_root,
        output_root=args.output_root,
        splits=[s.strip() for s in str(args.splits).split(",") if s.strip()],
        model=args.model,
        n_style=max(1, int(args.n_style)),
        api_base=args.api_base,
        force=bool(args.force),
        workers=max(1, int(args.workers)),
    )


if __name__ == "__main__":
    main()
