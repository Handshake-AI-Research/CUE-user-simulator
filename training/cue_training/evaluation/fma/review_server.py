"""Local FastAPI review UI for one FMA batch."""

from pathlib import Path
from typing import Any

from cue_training.evaluation.common.io import read_jsonl
from cue_training.evaluation.fma.paths import batch_dir, resolve_run_dir
from cue_training.evaluation.fma.review import (
    finish_batch,
    label_counts,
    merge_failure_modes,
    prune_empty_modes,
    rename_failure_mode,
)
from cue_training.evaluation.fma.store import load_taxonomy, save_taxonomy, upsert_mode

STATIC_DIR = Path(__file__).resolve().parent / "static"


def _enrich_taxonomy(
    run_dir: Path, taxonomy: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    counts = label_counts(run_dir)
    out = []
    for mode in taxonomy:
        row = dict(mode)
        row["n_samples"] = int(counts.get(str(mode.get("name") or ""), 0))
        out.append(row)
    return out


def _load_batch(run_dir: Path, batch: int) -> dict[str, Any]:
    bdir = batch_dir(run_dir, batch)
    sample = {str(r["primary_key"]): r for r in read_jsonl(str(bdir / "sample.jsonl"))}
    proposals = {
        str(r["primary_key"]): r for r in read_jsonl(str(bdir / "proposals.jsonl"))
    }
    decisions_path = bdir / "decisions.jsonl"
    decisions = {}
    if decisions_path.is_file():
        decisions = {str(r["primary_key"]): r for r in read_jsonl(str(decisions_path))}
    keys = list(sample.keys())
    taxonomy = load_taxonomy(run_dir)
    return {
        "keys": keys,
        "sample": sample,
        "proposals": proposals,
        "decisions": decisions,
        "taxonomy": _enrich_taxonomy(run_dir, taxonomy),
    }


def create_app(run_dir: Path, batch: int) -> Any:
    try:
        from fastapi import FastAPI, HTTPException
        from fastapi.responses import FileResponse
        from fastapi.staticfiles import StaticFiles
        from pydantic import BaseModel, Field
    except ImportError as exc:  # pragma: no cover
        raise ImportError(
            "fastapi is required for `cue-fma review`; install with: uv sync --group fma"
        ) from exc

    state = {"data": _load_batch(run_dir, batch)}

    class DecisionIn(BaseModel):
        primary_key: str
        label: str
        notes: str = ""
        description: str = ""
        explanation: str = ""
        turn_indices: list[int] | None = None
        excerpt: str = ""

    class FinishIn(BaseModel):
        decisions: list[DecisionIn] = Field(default_factory=list)

    class ModeIn(BaseModel):
        name: str
        description: str = ""

    class RenameIn(BaseModel):
        old: str
        new: str
        description: str = ""

    class MergeIn(BaseModel):
        sources: list[str]
        target: str
        description: str = ""

    app = FastAPI(title="FMA Review", docs_url=None, redoc_url=None)
    app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")

    @app.get("/")
    def index() -> FileResponse:
        return FileResponse(STATIC_DIR / "review.html")

    @app.get("/api/state")
    def api_state() -> dict[str, Any]:
        data = state["data"]
        items = []
        for key in data["keys"]:
            row = data["sample"][key]
            prop = data["proposals"].get(key) or {}
            dec = data["decisions"].get(key)
            items.append(
                {
                    "primary_key": key,
                    "episode_id": row.get("episode_id"),
                    "source_id": row.get("source_id"),
                    "source": row.get("source"),
                    "arm": row.get("arm"),
                    "variant": row.get("variant"),
                    "domain": row.get("domain"),
                    "task_id": row.get("task_id"),
                    "task_description": row.get("task_description"),
                    "success_criteria": row.get("success_criteria"),
                    "reward_report": row.get("reward_report"),
                    "conversation": row.get("conversation") or [],
                    "proposal": {
                        "label": prop.get("label"),
                        "explanation": prop.get("explanation"),
                        "turn_indices": prop.get("turn_indices") or [],
                        "excerpt": prop.get("excerpt") or "",
                    },
                    "decision": dec,
                }
            )
        return {
            "batch": batch,
            "run_dir": str(run_dir),
            "taxonomy": data["taxonomy"],
            "items": items,
        }

    @app.post("/api/mode")
    def api_mode(body: ModeIn) -> dict[str, Any]:
        name = body.name.strip()
        if not name:
            raise HTTPException(400, "name required")
        description = body.description.strip()
        taxonomy = load_taxonomy(run_dir)
        upsert_mode(taxonomy, name=name, description=description)
        # upsert_mode ignores empty descriptions; make the posted value authoritative
        # so the UI can also clear one.
        for mode in taxonomy:
            if mode.get("name") == name:
                mode["description"] = description
                break
        save_taxonomy(run_dir, taxonomy)
        state["data"]["taxonomy"] = _enrich_taxonomy(run_dir, taxonomy)
        return {"ok": True, "taxonomy": state["data"]["taxonomy"]}

    @app.post("/api/mode/rename")
    def api_rename(body: RenameIn) -> dict[str, Any]:
        old, new = body.old.strip(), body.new.strip()
        if not old or not new:
            raise HTTPException(400, "old and new required")
        result = rename_failure_mode(
            run_dir,
            old=old,
            new=new,
            description=body.description.strip() or None,
        )
        state["data"] = _load_batch(run_dir, batch)
        return {
            "ok": True,
            "taxonomy": state["data"]["taxonomy"],
            "remapped": result["remapped"],
        }

    @app.post("/api/mode/merge")
    def api_merge(body: MergeIn) -> dict[str, Any]:
        sources = [s.strip() for s in body.sources if s and s.strip()]
        target = body.target.strip()
        if not sources or not target:
            raise HTTPException(400, "sources and target required")
        result = merge_failure_modes(
            run_dir,
            sources=sources,
            target=target,
            description=body.description.strip() or None,
        )
        state["data"] = _load_batch(run_dir, batch)
        return {
            "ok": True,
            "taxonomy": state["data"]["taxonomy"],
            "remapped": result["remapped"],
        }

    @app.post("/api/mode/prune-empty")
    def api_prune_empty() -> dict[str, Any]:
        result = prune_empty_modes(run_dir)
        state["data"] = _load_batch(run_dir, batch)
        return {
            "ok": True,
            "taxonomy": state["data"]["taxonomy"],
            "removed": result["removed"],
        }

    @app.post("/api/finish")
    def api_finish(body: FinishIn) -> dict[str, Any]:
        if not body.decisions:
            raise HTTPException(400, "no decisions")
        result = finish_batch(
            run_dir,
            batch,
            [d.model_dump() for d in body.decisions],
        )
        state["data"] = _load_batch(run_dir, batch)
        return result

    return app


def serve(run: str, batch: int, *, host: str = "127.0.0.1", port: int = 8765) -> None:
    try:
        import uvicorn
    except ImportError as exc:  # pragma: no cover
        raise ImportError(
            "uvicorn is required for `cue-fma review`; install with: uv sync --group fma"
        ) from exc
    run_dir = resolve_run_dir(run)
    app = create_app(run_dir, batch)
    print(f"FMA review: http://{host}:{port}/  (run={run_dir} batch={batch})")
    uvicorn.run(app, host=host, port=port, log_level="info")
