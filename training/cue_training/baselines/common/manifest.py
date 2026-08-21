"""Fail-closed paper-parity baseline training manifests and artifact validation."""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
import tempfile
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

from cue_training.baselines.common.config import TrainConfig
from cue_training.runlog.log import log, warn


class TrainingError(RuntimeError):
    """Base class for paper-parity training failures."""


class UnsupportedTrainingError(TrainingError):
    pass


class MissingDataError(TrainingError):
    pass


class MissingGPUError(TrainingError):
    pass


class MissingCheckpointError(TrainingError):
    pass


class IncompleteArtifactError(TrainingError):
    pass


@dataclass(frozen=True)
class PaperStage:
    name: str
    description: str
    required_outputs: tuple[str, ...] = ()


@dataclass(frozen=True)
class PaperManifest:
    method: str
    preset: str
    stages: tuple[PaperStage, ...]
    model: str | None = None
    deviations: tuple[str, ...] = ()
    lean_preset: str | None = None
    requires_gpu: bool = False
    notes: str = ""


PAPER_MANIFESTS: dict[str, PaperManifest] = {
    "base": PaperManifest(
        method="base",
        preset="none",
        stages=(PaperStage("validate", "Validate configured API/local model."),),
        notes="No training; validate serving credentials/model id only.",
    ),
    "userlm": PaperManifest(
        method="userlm",
        preset="userlm_released",
        model="microsoft/UserLM-8b",
        stages=(
            PaperStage("record_model", "Record/download released UserLM-8b.", ("model_id.txt",)),
        ),
        notes="Inference-only; paper decoding temperature=1.0 top_p=0.8.",
    ),
    "realusersim": PaperManifest(
        method="realusersim",
        preset="realusersim_gpt4o_full",
        model="gpt-4o",
        stages=(
            PaperStage(
                "curate_wildchat",
                "WildChat-4.8M → English multi-turn GPT-4o trajs (~21.6k / ~7.3k users).",
            ),
            PaperStage(
                "extract_profiles",
                "GPT-4o Executable Persona Manual per unique user (~7.3k profiles).",
                ("profiles.jsonl",),
            ),
        ),
        notes=(
            "Eval samples the released Salesforce/RealUserSim profiles (as_is). "
            "Set REALUSERSIM_REBUILD=1 to instead re-curate WildChat-4.8M and re-extract."
        ),
    ),
    "usp": PaperManifest(
        method="usp",
        preset="usp_pretrained_hf",
        model="wangkevin02/USP",
        requires_gpu=False,
        lean_preset=None,
        stages=(
            PaperStage(
                "pretrained_hf",
                "Record the published HuggingFace USP checkpoint (no local training).",
                ("pretrained_model.txt",),
            ),
        ),
        notes="cue_training serves wangkevin02/USP via vLLM; SFT/RLCC are not shipped.",
    ),
    "ppol": PaperManifest(
        method="ppol",
        preset="ppol_openevolve",
        stages=(
            PaperStage("collect_baseline", "Collect baseline trajectories."),
            PaperStage("train_discriminator", "Train human-likeness discriminator."),
            PaperStage("evolve", "70-iteration OpenEvolve evolution.", ("best_program.py",)),
        ),
        notes="Uses a temporary generated OpenEvolve config rather than mutating vendored YAML.",
    ),
}


def manifest_path(artifacts_dir: Path) -> Path:
    return artifacts_dir / "manifest.json"


def _hash_path(path: Path | None) -> str | None:
    if path is None or not Path(path).exists():
        return None
    p = Path(path)
    if p.is_dir():
        return f"dir:{p.resolve()}"
    h = hashlib.sha256()
    with p.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()[:16]


def write_training_manifest(
    artifacts_dir: Path,
    *,
    paper: PaperManifest,
    status: str,
    stages_done: list[str],
    data_path: Path | None = None,
    world_size: int = 1,
    deviations: list[str] | None = None,
    extra: dict[str, Any] | None = None,
) -> Path:
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "method": paper.method,
        "preset": paper.preset,
        "lean_preset": paper.lean_preset,
        "model": paper.model,
        "status": status,
        "stages": [asdict(s) for s in paper.stages],
        "stages_done": stages_done,
        "data_hash": _hash_path(data_path),
        "model_hash": paper.model,
        "deviations": list(deviations or paper.deviations),
        "gpu_world_size": world_size,
        "updated_at": datetime.now(timezone.utc).isoformat(),
        **(extra or {}),
    }
    path = manifest_path(artifacts_dir)
    fd, tmp = tempfile.mkstemp(prefix="manifest.", dir=str(artifacts_dir))
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True)
            handle.write("\n")
        os.replace(tmp, path)
    finally:
        if os.path.exists(tmp):
            try:
                os.unlink(tmp)
            except OSError:
                pass
    return path


def load_training_manifest(artifacts_dir: Path) -> dict[str, Any] | None:
    path = manifest_path(artifacts_dir)
    if not path.is_file():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def validate_artifacts(method: str, artifacts_dir: Path, *, require_complete: bool = True) -> dict[str, Any]:
    paper = PAPER_MANIFESTS.get(method)
    if paper is None:
        raise UnsupportedTrainingError(f"no paper manifest for {method!r}")
    artifacts_dir = Path(artifacts_dir)
    if method == "base":
        return {"method": method, "status": "ok", "preset": "none"}

    manifest = load_training_manifest(artifacts_dir)
    if require_complete and (manifest is None or manifest.get("status") != "complete"):
        raise IncompleteArtifactError(
            f"{method}: missing complete manifest.json under {artifacts_dir}"
        )
    if manifest is not None and manifest.get("preset") not in {paper.preset, paper.lean_preset}:
        raise IncompleteArtifactError(
            f"{method}: manifest preset {manifest.get('preset')!r} incompatible with "
            f"paper preset {paper.preset!r}"
        )
    missing: list[str] = []
    for stage in paper.stages:
        for rel in stage.required_outputs:
            # Accept either exact relative path or common alternate names.
            candidates = [artifacts_dir / rel]
            if rel == "adapter":
                candidates += [artifacts_dir / "adapter", artifacts_dir / "sft_adapter"]
            if rel == "sft_adapter":
                candidates += [artifacts_dir / "sft_adapter", artifacts_dir / "adapter"]
            if rel == "rlcc_adapter":
                candidates.append(artifacts_dir / "rlcc_adapter")
            if rel == "grpo_adapter":
                candidates.append(artifacts_dir / "grpo_adapter")
            if rel == "profiles.jsonl":
                candidates.append(artifacts_dir / "profiles.jsonl")
            if rel == "best_program.py":
                candidates += [
                    artifacts_dir / "best_program.py",
                    artifacts_dir.parent / "best_program.py",
                ]
            if not any(c.exists() for c in candidates):
                missing.append(rel)
            elif rel in ("profiles.jsonl", "best_program.py"):
                # A zero-byte stub satisfies existence but has no usable content.
                if not any(c.is_file() and c.stat().st_size > 0 for c in candidates):
                    missing.append(rel)
    if missing and require_complete:
        raise IncompleteArtifactError(
            f"{method}: incomplete artifacts under {artifacts_dir}; missing {missing}"
        )
    return manifest or {"method": method, "status": "validated"}


def _require_data(path: Path | None) -> Path:
    if path is None or not Path(path).exists():
        raise MissingDataError(f"training data path missing: {path}")
    return Path(path)


def _require_gpu(method: str) -> None:
    try:
        import torch
    except ImportError as exc:
        raise MissingGPUError(f"{method} paper training requires torch+GPU") from exc
    if not torch.cuda.is_available():
        raise MissingGPUError(f"{method} paper training requires a CUDA GPU")


def train_paper(
    method: str,
    *,
    artifact_root: Path,
    data_path: Path | None,
    jsonl_names: tuple[str, ...] = ("train.jsonl",),
    force: bool = False,
    dry_run: bool = False,
    paper_overrides: dict[str, Any] | None = None,
    repo_root: Path | None = None,
) -> Path:
    """Run fail-closed paper-parity training for one baseline method."""

    if method not in PAPER_MANIFESTS:
        raise UnsupportedTrainingError(f"unsupported training method {method!r}")
    paper = PAPER_MANIFESTS[method]
    overrides = dict(paper_overrides or {})
    if method == "ppol":
        # Keep model-tagged subdir compatibility with scripts/train_ppol.sh defaults.
        domain = str(overrides.get("domain") or "retail_airline")
        sim_models = overrides.get("sim_models")
        if isinstance(sim_models, (list, tuple)) and sim_models:
            sims = [str(s) for s in sim_models if str(s).strip()]
        else:
            sims = [str(overrides.get("sim_model") or "gpt-5.4-mini")]
        artifact_dirs = [
            Path(artifact_root)
            / "ppol"
            / f"{domain}_{s.replace('/', '__').replace(':', '__')}"
            for s in sims
        ]
        artifacts = artifact_dirs[0]

        if not force:
            try:
                for d in artifact_dirs:
                    validate_artifacts(method, d, require_complete=True)
                log(
                    "baselines.manifest",
                    f"{method}: complete artifacts already present for {len(artifact_dirs)} sim(s)",
                )
                return artifacts
            except (IncompleteArtifactError, FileNotFoundError):
                pass

        if dry_run:
            log(
                "baselines.manifest",
                f"dry-run train_paper {method} preset={paper.preset} -> {len(artifact_dirs)} sim(s)",
            )
            for d in artifact_dirs:
                write_training_manifest(
                    d,
                    paper=paper,
                    status="dry_run",
                    stages_done=[],
                    data_path=data_path,
                    deviations=["dry_run"],
                )
            return artifacts

        root = repo_root or Path.cwd()
        script = root / "scripts" / "train_ppol.sh"
        if not script.is_file():
            raise MissingCheckpointError(f"ppol training script missing: {script}")
        env = dict(os.environ)
        env["PPOL_ITERATIONS"] = str(overrides.get("iterations") or 70)
        env["PPOL_DOMAIN"] = domain
        env["PPOL_AGENT_MODEL"] = str(overrides.get("agent_model") or "gpt-5.2")
        env["PPOL_GEN_MODEL"] = str(
            overrides.get("gen_model")
            or overrides.get("mutation_model")
            or "openrouter/google/gemini-3-flash-preview"
        )
        env["PPOL_MUTATION_MODEL"] = str(
            overrides.get("mutation_model") or env["PPOL_GEN_MODEL"]
        )
        # Paper curriculum N=5→8→10 is the vendored PersonaPoliciesConfig default.
        env.setdefault("PERSONA_POLICIES_CURRICULUM", "1")
        if len(sims) == 1:
            env["PPOL_SINGLE"] = "1"
            env["PPOL_SIM_MODEL"] = sims[0]
            env["ARTIFACTS_DIR"] = str(artifact_dirs[0])
            env["PPOL_BEST_PROGRAM"] = str(artifact_dirs[0] / "best_program.py")
        else:
            env.pop("PPOL_SINGLE", None)
            env["PPOL_SIM_MODELS"] = ",".join(sims)
            env.pop("ARTIFACTS_DIR", None)
            env.pop("PPOL_BEST_PROGRAM", None)
        # Prefer temporary generated OpenEvolve config via env when the script supports it.
        env["PPOL_OPENEVOLVE_CONFIG_MODE"] = "temp"
        if overrides.get("hosted_vllm_api_base"):
            env["HOSTED_VLLM_API_BASE"] = str(overrides["hosted_vllm_api_base"])
            env.setdefault("HOSTED_VLLM_API_KEY", os.environ.get("HOSTED_VLLM_API_KEY", "EMPTY"))
        log("baselines.manifest", f"ppol: invoking {script} for {len(sims)} sim(s)")
        subprocess.run(["bash", str(script)], check=True, env=env, cwd=str(root))  # noqa: S603
        for d in artifact_dirs:
            if not (d / "best_program.py").is_file():
                raise IncompleteArtifactError(f"ppol did not produce {d / 'best_program.py'}")
            write_training_manifest(
                d,
                paper=paper,
                status="complete",
                stages_done=["collect_baseline", "train_discriminator", "evolve"],
            )
        return artifacts

    artifacts = Path(artifact_root) / method
    if not force:
        try:
            validate_artifacts(method, artifacts, require_complete=True)
            log("baselines.manifest", f"{method}: complete artifacts already present at {artifacts}")
            return artifacts
        except (IncompleteArtifactError, FileNotFoundError):
            pass

    if dry_run:
        log("baselines.manifest", f"dry-run train_paper {method} preset={paper.preset} -> {artifacts}")
        write_training_manifest(
            artifacts,
            paper=paper,
            status="dry_run",
            stages_done=[],
            data_path=data_path,
            deviations=["dry_run"],
        )
        return artifacts

    stages_done: list[str] = []
    if method == "base":
        write_training_manifest(artifacts, paper=paper, status="complete", stages_done=["validate"])
        return artifacts

    if method == "userlm":
        model = paper.model or "microsoft/UserLM-8b"
        artifacts.mkdir(parents=True, exist_ok=True)
        (artifacts / "model_id.txt").write_text(model + "\n", encoding="utf-8")
        # Optional download via huggingface_hub when available.
        try:
            from huggingface_hub import snapshot_download

            snapshot_download(repo_id=model, local_files_only=False)
        except Exception as exc:  # noqa: BLE001
            warn("baselines.manifest", f"userlm download skipped/failed: {exc}")
        stages_done.append("record_model")
        write_training_manifest(artifacts, paper=paper, status="complete", stages_done=stages_done)
        return artifacts

    if method == "realusersim":
        # Paper train: WildChat-4.8M → filter → per-user GPT-4o personas (App. A).
        # Eval is as_is pool sampling from that artifact (not τ-USI paired extraction).
        from cue_training.baselines.realusersim.baseline import RealUserSimBaseline
        from cue_training.baselines.realusersim.profiles import PROFILES_FILE

        use_published = not bool(
            overrides.get("rebuild_from_wildchat")
            or overrides.get("use_published_profiles") is False
            or str(os.environ.get("REALUSERSIM_REBUILD", "")).lower()
            in {"1", "true", "yes"}
        )
        cfg = TrainConfig(
            output_dir=artifacts,
            data_path=Path(data_path) if data_path is not None else Path("."),
            sim_model=str(overrides.get("profiler_model") or paper.model or "gpt-4o"),
            api_key_env=str(overrides.get("api_key_env") or "OPENAI_API_KEY"),
            api_base=overrides.get("api_base"),
            limit=overrides.get("limit"),  # None = all users
            batch_size=int(overrides.get("batch_size") or 32),
            jsonl_names=jsonl_names,
            extra={
                "paper_parity": True,
                "use_published_profiles": use_published,
                "rebuild_from_wildchat": not use_published,
                "skip_curate": bool(overrides.get("skip_curate")),
                "wildchat_repo": overrides.get("wildchat_repo"),
                "max_rows": overrides.get("max_rows"),
                "legacy_per_conversation": bool(overrides.get("legacy_per_conversation")),
                "streaming": overrides.get("streaming", True),
            },
        )
        RealUserSimBaseline().train(cfg)
        stages_done.append(
            "import_published_profiles" if use_published else "extract_profiles_wildchat"
        )
        write_training_manifest(
            artifacts,
            paper=paper,
            status="complete",
            stages_done=stages_done,
            data_path=None if use_published else data_path,
            deviations=(
                [] if use_published else ["rebuilt_from_wildchat_not_published"]
            ),
        )
        if not (artifacts / PROFILES_FILE).is_file():
            raise IncompleteArtifactError(f"realusersim missing {artifacts / PROFILES_FILE}")
        return artifacts

    if method == "usp":
        # Published HF checkpoint only — no local SFT/RLCC in cue_training.
        artifacts.mkdir(parents=True, exist_ok=True)
        pretrained = str(
            overrides.get("pretrained_model")
            or overrides.get("model")
            or "wangkevin02/USP"
        )
        (artifacts / "pretrained_model.txt").write_text(pretrained + "\n", encoding="utf-8")
        write_training_manifest(
            artifacts,
            paper=paper,
            status="complete",
            stages_done=["pretrained_hf"],
            data_path=None,
        )
        return artifacts

    raise UnsupportedTrainingError(f"no train_paper implementation for {method}")
