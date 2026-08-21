"""Lazy access to an external SimulatorArena checkout."""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

_HERE = Path(__file__).resolve().parent
_SUBMODULE = _HERE / "external" / "SimulatorArena"


def repo_path() -> Path:
    env = os.environ.get("SIMULATORARENA_PATH")
    if env:
        return Path(env)
    if (_SUBMODULE / "simulation").exists():
        return _SUBMODULE
    return _HERE.parents[4] / "SimulatorArena"


def ensure_importable(path: Path | None = None) -> Path:
    repo = Path(path) if path else repo_path()
    sim = repo / "simulation"
    if not sim.exists():
        raise FileNotFoundError(
            f"SimulatorArena not found at {sim}. Clone it under "
            "cue_training/evaluation/simulatorarena/external/SimulatorArena "
            "or set SIMULATORARENA_PATH."
        )
    for sub in (str(sim), str(repo / "evaluation")):
        if sub not in sys.path:
            sys.path.insert(0, sub)
    return repo


def commit(repo: Path) -> str | None:
    try:
        out = subprocess.run(["git", "-C", str(repo), "rev-parse", "HEAD"], capture_output=True, text=True, check=False)
        return out.stdout.strip() or None
    except Exception:  # noqa: BLE001
        return None


def assistant_reply(
    model: str, messages: list[dict[str, str]], *, temperature: float = 0.0, max_tokens: int = 1024
) -> str:
    """Produce one assistant turn.

    Reuses the CUE/baseline LiteLLM client (same provider surface SimulatorArena uses
    for API models); for local assistants the SimulatorArena vLLM path can be swapped in
    here during execution. Kept isolated so the assistant is consistent across variants.
    """

    from cue_training.baselines.common.llm import chat
    from cue_training.evaluation.common.dataset_assistants import api_key_env_for_model

    return chat(
        messages,
        model=model,
        api_key_env=api_key_env_for_model(model),
        temperature=temperature,
        max_tokens=max_tokens,
    )


def run_eval_script(repo: Path, task_subdir: str, script: str, file_name: str, annotation_id: str) -> int:
    """Invoke a SimulatorArena evaluation shell script (best-effort passthrough)."""

    cwd = repo / "evaluation" / task_subdir
    cmd = ["bash", f"./{script}", "--file_name", file_name, "--annotation_id", annotation_id]
    proc = subprocess.run(cmd, cwd=str(cwd), check=False)
    return proc.returncode
