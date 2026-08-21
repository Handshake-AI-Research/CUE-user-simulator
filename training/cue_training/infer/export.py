"""Bulk trajectory-embedding export for evaluation harnesses."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Iterable, Iterator

import torch

from cue_training.data.schema import validate_record
from cue_training.infer import load_unified
from cue_training.infer.session_preprocess import (
    normalize_session_preprocess,
    preprocess_turns,
)

_CONVERSATION_KEYS = ("real_conversation", "conversation", "turns")


def normalize_export_record(record: Any) -> dict[str, Any]:
    """Normalize an evaluation record into session id, turns, and metadata."""

    if not isinstance(record, dict):
        raise ValueError("record must be an object")
    session_id = str(
        record.get("session_id") or record.get("id") or record.get("episode_id") or ""
    ).strip()
    if not session_id:
        raise ValueError("record requires session_id, id, or episode_id")
    turns = next(
        (
            record[key]
            for key in _CONVERSATION_KEYS
            if isinstance(record.get(key), list) and record[key]
        ),
        None,
    )
    if turns is None:
        raise ValueError(
            "record requires a nonempty real_conversation, conversation, or turns list"
        )
    normalized = validate_record({"session_id": session_id, "turns": turns})
    metadata = record.get("metadata")
    return {
        "session_id": session_id,
        "turns": normalized["turns"],
        "metadata": dict(metadata) if isinstance(metadata, dict) else {},
    }


def iter_export_records(path: str | Path) -> Iterator[dict[str, Any]]:
    path = Path(path)
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                yield normalize_export_record(json.loads(line))
            except (json.JSONDecodeError, ValueError) as exc:
                raise ValueError(f"{path}:{line_number}: {exc}") from exc


def load_checkpoint(checkpoint: str | Path, *, device: str = "cpu") -> Any:
    """Load a unified joint training or decoder refinement checkpoint."""

    checkpoint = Path(checkpoint)
    from cue_training.refinement.train import load_refinement_model, refinement_config_path

    if refinement_config_path(checkpoint) is not None:
        return load_refinement_model(checkpoint, device=device)
    if (checkpoint / "unified_config.json").is_file():
        return load_unified(checkpoint, device=device)
    if (checkpoint / "config.json").is_file():
        from transformers import AutoModel

        model = AutoModel.from_pretrained(
            str(checkpoint), trust_remote_code=True,
            torch_dtype=torch.bfloat16 if str(device).startswith("cuda") else None,
        ).to(device)
        return model.eval()
    raise ValueError(
        f"{checkpoint} is not a joint training or decoder refinement unified checkpoint"
    )


def export_embeddings(
    checkpoint: str | Path,
    input_path: str | Path,
    output_path: str | Path,
    *,
    device: str = "cpu",
    batch_size: int = 16,
    session_preprocess: str = "full",
) -> Path:
    """Encode normalized JSONL records and atomically write harness rows."""

    if batch_size < 1:
        raise ValueError("batch_size must be positive")
    mode = normalize_session_preprocess(session_preprocess)
    model = load_checkpoint(checkpoint, device=device)
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    # Pid-scoped tmp so concurrent seed workers do not unlink each other's file.
    temporary = output.with_name(f".{output.name}.tmp.{os.getpid()}")
    batch: list[dict[str, Any]] = []
    try:
        with temporary.open("w", encoding="utf-8") as handle:
            for record in iter_export_records(input_path):
                batch.append(record)
                if len(batch) >= batch_size:
                    _write_batch(model, batch, handle, session_preprocess=mode)
                    batch.clear()
            if batch:
                _write_batch(model, batch, handle, session_preprocess=mode)
        os.replace(temporary, output)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise
    return output


def export_embedding_records(
    checkpoint: str | Path,
    records: Iterable[dict[str, Any]],
    output_path: str | Path,
    *,
    device: str = "cpu",
    batch_size: int = 16,
    session_preprocess: str = "full",
) -> Path:
    """Encode in-memory evaluation rows and atomically write harness embeddings."""

    if batch_size < 1:
        raise ValueError("batch_size must be positive")
    mode = normalize_session_preprocess(session_preprocess)
    model = load_checkpoint(checkpoint, device=device)
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    # Pid-scoped tmp so concurrent seed workers do not unlink each other's file.
    temporary = output.with_name(f".{output.name}.tmp.{os.getpid()}")
    batch: list[dict[str, Any]] = []
    try:
        with temporary.open("w", encoding="utf-8") as handle:
            for raw in records:
                batch.append(normalize_export_record(raw))
                if len(batch) >= batch_size:
                    _write_batch(model, batch, handle, session_preprocess=mode)
                    batch.clear()
            if batch:
                _write_batch(model, batch, handle, session_preprocess=mode)
        os.replace(temporary, output)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise
    return output


def _write_batch(
    model: Any,
    batch: list[dict[str, Any]],
    handle: Any,
    *,
    session_preprocess: str = "full",
) -> None:
    sessions = [
        preprocess_turns(record["turns"], session_preprocess) for record in batch
    ]
    with torch.no_grad():
        embeddings = model.encode(sessions)
    if len(embeddings) != len(batch):
        raise ValueError(
            f"encoder returned {len(embeddings)} embeddings for {len(batch)} records"
        )
    for record, embedding in zip(batch, embeddings, strict=True):
        row = {
            "trajectory_embedding": embedding.detach().float().cpu().tolist(),
            "session_id": record["session_id"],
            "metadata": record["metadata"],
        }
        handle.write(json.dumps(row, ensure_ascii=False) + "\n")
