"""Thin lazy wrapper around tau2-bench: import, resolve tasks, run, read reward."""

from __future__ import annotations

import os
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

_HERE = Path(__file__).resolve().parent
# External checkout location, with env override + sibling-clone fallback for dev.
_SUBMODULE = _HERE / "external" / "tau2-bench"


def tau2_repo_path() -> Path:
    env = os.environ.get("TAU2_BENCH_PATH")
    if env:
        return Path(env)
    if (_SUBMODULE / "src").exists():
        return _SUBMODULE
    sibling = _HERE.parents[4] / "tau2-bench"
    return sibling


def ensure_tau2_importable(repo_path: Path | None = None) -> Path:
    repo = Path(repo_path) if repo_path else tau2_repo_path()
    src = repo / "src"
    if not src.exists():
        raise FileNotFoundError(
            f"tau2-bench source not found at {src}. Clone it under "
            "cue_training/evaluation/tau2_bench/external/tau2-bench "
            "or set TAU2_BENCH_PATH."
        )
    if str(src) not in sys.path:
        sys.path.insert(0, str(src))
    return repo


_console_quieted = False


def quiet_tau2_console(repo_path: Path | None = None) -> None:
    """Silence tau2's per-episode terminal noise so our progress bar stays readable.

    Two sources: ``console_display=False`` only hides the simulation transcript, while
    per-task banners and ``try_resume``'s "Resuming run from N runs" print regardless;
    and ``get_info`` shells out to ``git rev-parse HEAD`` in the process cwd, which leaks
    git stderr plus a loguru ERROR per episode when cwd is not a repo. Real failures still
    surface via loguru and our own result rows.
    """

    global _console_quieted
    if _console_quieted:
        return
    from rich.console import Console  # type: ignore
    from tau2.runner import helpers  # type: ignore
    from tau2.utils.display import ConsoleDisplay  # type: ignore

    ConsoleDisplay.console = Console(quiet=True)
    commit = (tau2_commit(repo_path) if repo_path else None) or "unknown"
    helpers.get_commit_hash = lambda: commit
    _console_quieted = True


def tau2_commit(repo_path: Path) -> str | None:
    try:
        out = subprocess.run(
            ["git", "-C", str(repo_path), "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            check=False,
        )
        return out.stdout.strip() or None
    except Exception:  # noqa: BLE001
        return None


@dataclass
class Tau2Settings:
    repo_path: Path
    output_dir: Path
    agent: str = "llm_agent"
    user: str = "cue_eval_user"
    agent_llm: str = "gpt-5.2"
    agent_reasoning_effort: str | None = "high"
    max_steps: int = 100
    max_concurrency: int = 1
    seed: int = 300


@dataclass
class Tau2Result:
    task_id: str
    domain: str
    found_task: bool
    reward: float | None = None
    is_success: bool | None = None
    termination_reason: str | None = None
    num_turns: int | None = None
    conversation: list[dict[str, Any]] = field(default_factory=list)
    # Failed programmatic checks only (action / nl / communicate / db). Metrics still
    # use scalar reward/is_success; FMA uses this for grounding.
    reward_report: dict[str, Any] | None = None
    save_path: str | None = None
    error: str | None = None


def resolve_task(get_tasks: Any, domain: str, task_id: str) -> Any | None:
    try:
        tasks = list(get_tasks(domain, task_ids=[str(task_id)]))
    except Exception:  # noqa: BLE001
        tasks = []
    if tasks:
        return tasks[0]
    try:
        for task in get_tasks(domain, task_ids=None):
            if str(task.id) == str(task_id):
                return task
    except Exception:  # noqa: BLE001
        return None
    return None


def _jsonable(value: Any) -> Any:
    """Best-effort conversion of pydantic / nested objects for JSONL export."""

    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    model_dump = getattr(value, "model_dump", None)
    if callable(model_dump):
        try:
            return _jsonable(model_dump(mode="json"))
        except TypeError:
            return _jsonable(model_dump())
    as_dict = getattr(value, "dict", None)
    if callable(as_dict):
        return _jsonable(as_dict())
    return str(value)


def _serialize_tool_call(tc: Any) -> dict[str, Any]:
    return {
        "id": str(getattr(tc, "id", "") or ""),
        "name": str(getattr(tc, "name", "") or ""),
        "arguments": _jsonable(getattr(tc, "arguments", None) or {}),
        "requestor": str(getattr(tc, "requestor", "") or "assistant"),
    }


def _message_dicts(simulation: Any) -> list[dict[str, Any]]:
    """Project tau2 messages to JSONL turns, keeping full tool calls and tool results.

    Metrics consumers may ignore ``tool_calls`` / tool-role rows; FMA uses them.
    """

    out: list[dict[str, Any]] = []
    for msg in getattr(simulation, "messages", []) or []:
        role = str(getattr(msg, "role", None) or "")
        content = getattr(msg, "content", None)
        entry: dict[str, Any] = {"role": role, "content": content or ""}
        tool_calls = getattr(msg, "tool_calls", None)
        if tool_calls:
            entry["tool_calls"] = [_serialize_tool_call(tc) for tc in tool_calls]
        if role == "tool":
            entry["tool_call_id"] = str(getattr(msg, "id", "") or "")
            if getattr(msg, "requestor", None) is not None:
                entry["requestor"] = str(msg.requestor)
            if bool(getattr(msg, "error", False)):
                entry["error"] = True
        if getattr(msg, "raw_data", None):  # per-turn provenance (e.g. hard-decoder command block)
            entry["raw_data"] = msg.raw_data
        out.append(entry)
    return out


def reward_report_from_info(info: Any) -> dict[str, Any] | None:
    """Slim failure report from tau2 ``RewardInfo`` (failed checks only)."""

    if info is None:
        return None
    failed: list[dict[str, Any]] = []

    db = getattr(info, "db_check", None)
    if db is not None and getattr(db, "db_match", True) is False:
        failed.append(
            {
                "kind": "db",
                "passed": False,
                "db_reward": float(getattr(db, "db_reward", 0.0) or 0.0),
            }
        )

    for check in getattr(info, "action_checks", None) or []:
        if getattr(check, "action_match", True):
            continue
        action = getattr(check, "action", None)
        failed.append(
            {
                "kind": "action",
                "passed": False,
                "name": str(getattr(action, "name", "") or "") if action is not None else "",
                "arguments": _jsonable(getattr(action, "arguments", None) or {}) if action is not None else {},
                "action_id": str(getattr(action, "action_id", "") or "") if action is not None else "",
                "action_reward": float(getattr(check, "action_reward", 0.0) or 0.0),
                "tool_type": str(getattr(check, "tool_type", "") or "") or None,
            }
        )

    for check in getattr(info, "nl_assertions", None) or []:
        if getattr(check, "met", True):
            continue
        failed.append(
            {
                "kind": "nl_assertion",
                "passed": False,
                "nl_assertion": str(getattr(check, "nl_assertion", "") or ""),
                "justification": str(getattr(check, "justification", "") or ""),
            }
        )

    for check in getattr(info, "communicate_checks", None) or []:
        if getattr(check, "met", True):
            continue
        failed.append(
            {
                "kind": "communicate",
                "passed": False,
                "info": str(getattr(check, "info", "") or ""),
                "justification": str(getattr(check, "justification", "") or ""),
            }
        )

    for check in getattr(info, "env_assertions", None) or []:
        if getattr(check, "met", True):
            continue
        failed.append(
            {
                "kind": "env_assertion",
                "passed": False,
                "env_assertion": _jsonable(getattr(check, "env_assertion", None)),
                "reward": float(getattr(check, "reward", 0.0) or 0.0),
            }
        )

    breakdown = getattr(info, "reward_breakdown", None)
    basis = getattr(info, "reward_basis", None)
    report: dict[str, Any] = {
        "reward": float(info.reward) if getattr(info, "reward", None) is not None else None,
        "failed": failed,
    }
    if breakdown:
        report["reward_breakdown"] = {str(getattr(k, "value", k)): float(v) for k, v in dict(breakdown).items()}
    if basis:
        report["reward_basis"] = [str(getattr(x, "value", x)) for x in list(basis)]
    return report


def run_single_task(
    settings: Tau2Settings,
    *,
    domain: str,
    task_id: str,
    instructions: str,
    llm_args_user: dict[str, Any],
    num_trials: int,
    run_label: str,
) -> Tau2Result:
    """Run one tau2 task closed-loop with our registered user + conditioning."""

    ensure_tau2_importable(settings.repo_path)
    quiet_tau2_console(settings.repo_path)
    from tau2.data_model.simulation import TextRunConfig  # type: ignore
    from tau2.runner import get_tasks  # type: ignore
    from tau2.runner.batch import run_tasks  # type: ignore

    try:
        from tau2.metrics.agent_metrics import is_successful  # type: ignore
    except Exception:  # noqa: BLE001
        is_successful = None  # type: ignore

    task = resolve_task(get_tasks, domain, task_id)
    if task is None:
        return Tau2Result(
            task_id=str(task_id),
            domain=domain,
            found_task=False,
            error=f"task '{task_id}' not in tau2 domain '{domain}'",
        )
    # WHAT (the user's goal) comes from tau2's own user_scenario -- it is authoritative
    # for the reward. HOW (style/persona) is injected via llm_args_user. We do not
    # overwrite the scenario; `instructions` is retained only as an llm_args fallback.

    run_dir = Path(settings.output_dir) / run_label
    run_dir.mkdir(parents=True, exist_ok=True)
    save_path = run_dir / f"{domain}_{task_id}.json"

    config = TextRunConfig(
        domain=domain,
        agent=settings.agent,
        user=settings.user,
        llm_agent=settings.agent_llm,
        llm_args_agent={"reasoning_effort": settings.agent_reasoning_effort} if settings.agent_reasoning_effort else {},
        llm_user=str(llm_args_user.get("variant", "cue_eval")),
        llm_args_user=llm_args_user,
        num_trials=int(num_trials),
        max_steps=settings.max_steps,
        max_concurrency=max(settings.max_concurrency, 1),
        seed=settings.seed,
        save_to=str(save_path),
        # Never prompt on an existing save file: parallel / non-interactive runners have
        # no stdin, so tau2's resume/overwrite input() would raise EOFError. auto_resume
        # reuses a completed save and otherwise re-runs, without prompting.
        auto_resume=True,
    )
    results = run_tasks(config, [task], save_path=save_path, console_display=False)
    simulations = list(getattr(results, "simulations", []) or [])
    if not simulations:
        return Tau2Result(
            task_id=str(task_id),
            domain=domain,
            found_task=True,
            error="tau2 produced no simulations",
            save_path=str(save_path),
        )

    rewards = []
    for sim in simulations:
        info = getattr(sim, "reward_info", None)
        if info is not None and getattr(info, "reward", None) is not None:
            rewards.append(float(info.reward))
    mean_reward = sum(rewards) / len(rewards) if rewards else None
    last = simulations[-1]
    success: bool | None = None
    if mean_reward is not None:
        success = bool(is_successful(mean_reward)) if is_successful else mean_reward >= 1.0 - 1e-6

    return Tau2Result(
        task_id=str(task_id),
        domain=domain,
        found_task=True,
        reward=mean_reward,
        is_success=success,
        termination_reason=str(getattr(last, "termination_reason", "") or "") or None,
        num_turns=len(getattr(last, "messages", []) or []),
        conversation=_message_dicts(last),
        reward_report=reward_report_from_info(getattr(last, "reward_info", None)),
        save_path=str(save_path),
    )
