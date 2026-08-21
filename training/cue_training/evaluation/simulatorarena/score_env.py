"""Score every SimulatorArena rollout with paper math-correctness + writing 1–10 ratings.

Discovers ``rollout.simulatorarena.jsonl`` under a rollouts tree, rebuilds native JSON,
converts it to SimulatorArena's simulation-output schema, and runs the vendored OpenAI
batch scripts:

  math     — answer extraction, then Correct/Incorrect vs gold
  writing  — document extraction, document rating (1–10), interaction rating (1–10)

    uv run --with openai --with python-dotenv simarena-score \\
        --rollouts-root "$CUE_STORAGE_ROOT/outputs/rollouts"

Requires OPENAI_API_KEY. Batches poll until complete (default 30s).
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from tqdm import tqdm

from cue_training.utils.config import storage_root
from cue_training.evaluation.simulatorarena.bridge import repo_path
from cue_training.evaluation.simulatorarena.runner import _rebuild_native_from_rollouts
from cue_training.runlog.log import log, warn

TAG = "simarena-score"

MATH_ANN = "math_tutoring_annotations"
WRITE_ANN = "document_creation_annotations"
_ROLLOUT_NAME = "rollout.simulatorarena.jsonl"


@dataclass(frozen=True)
class NativeJob:
    domain: str  # math | writing
    job_dir: Path
    native_path: Path
    file_name: str


def _turns(record: dict[str, Any]) -> list[dict[str, str]]:
    conv = record.get("conversation") or record.get("assistant_messages") or []
    out: list[dict[str, str]] = []
    for turn in conv:
        if isinstance(turn, (list, tuple)) and len(turn) >= 2:
            out.append({"role": str(turn[0]), "content": str(turn[1] or "")})
        elif isinstance(turn, dict):
            out.append(
                {
                    "role": str(turn.get("role") or ""),
                    "content": str(turn.get("content") or ""),
                }
            )
    return out


def load_math_questions(repo: Path) -> dict[str, str]:
    path = repo / "data" / "math_tutoring_annotations.json"
    if not path.is_file():
        raise FileNotFoundError(
            f"Missing {path}. Restore MATH annotations (see SimulatorArena/data/README.md)."
        )
    rows = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(rows, list):
        rows = list(rows.values())
    out: dict[str, str] = {}
    for row in rows:
        pid = row.get("problem_id")
        q = row.get("question")
        if pid is None or not q:
            continue
        out[str(pid)] = str(q)
    return out


def to_paper_math(native: dict[str, Any], questions: dict[str, str]) -> dict[str, Any]:
    paper: dict[str, Any] = {}
    for model, problems in native.items():
        if not isinstance(problems, dict):
            continue
        for pid, workers in problems.items():
            if not isinstance(workers, dict):
                continue
            question = questions.get(str(pid), "")
            for worker, rec in workers.items():
                if not isinstance(rec, dict):
                    continue
                msgs = _turns(rec)
                first = next((m["content"] for m in msgs if m["role"] == "user"), "")
                queries = rec.get("user_queries") or [
                    m["content"] for m in msgs if m["role"] == "user"
                ]
                paper.setdefault(str(model), {}).setdefault(str(pid), {})[str(worker)] = {
                    "assistant_messages": msgs,
                    "first_query_content": str(queries[0] if queries else first),
                    "problem": question,
                    "user_queries": list(queries),
                    "ai_responses": rec.get("ai_responses")
                    or [m["content"] for m in msgs if m["role"] == "assistant"],
                }
    return paper


def to_paper_writing(native: dict[str, Any]) -> dict[str, Any]:
    paper: dict[str, Any] = {}
    for model, docs in native.items():
        if not isinstance(docs, dict):
            continue
        for doc_type, intents in docs.items():
            if not isinstance(intents, dict):
                continue
            for intent, workers in intents.items():
                if not isinstance(workers, dict):
                    continue
                for worker, rec in workers.items():
                    if not isinstance(rec, dict):
                        continue
                    msgs = _turns(rec)
                    paper.setdefault(str(model), {}).setdefault(str(doc_type), {}).setdefault(
                        str(intent), {}
                    )[str(worker)] = {
                        "assistant_messages": msgs,
                        "user_queries": rec.get("user_queries")
                        or [m["content"] for m in msgs if m["role"] == "user"],
                        "ai_responses": rec.get("ai_responses")
                        or [m["content"] for m in msgs if m["role"] == "assistant"],
                    }
    return paper


def _domain_from_path(path: Path) -> str | None:
    parts = path.parts
    if "simulatorarena" not in parts:
        return None
    i = parts.index("simulatorarena")
    if i + 1 >= len(parts):
        return None
    domain = parts[i + 1]
    if domain in {"math", "writing"}:
        return domain
    return None


def _file_name(rollouts_root: Path, job_dir: Path, native_stem: str) -> str:
    try:
        rel = job_dir.resolve().relative_to(rollouts_root.resolve())
    except ValueError:
        rel = Path(job_dir.name)
    slug = "__".join(rel.parts + (native_stem,))
    return slug.replace(" ", "_")


def discover_jobs(rollouts_root: Path, *, domains: set[str]) -> list[NativeJob]:
    jobs: list[NativeJob] = []
    for rollout in sorted(rollouts_root.rglob(_ROLLOUT_NAME)):
        domain = _domain_from_path(rollout)
        if domain is None or domain not in domains:
            continue
        job_dir = rollout.parent
        native_dir = job_dir / "native"
        if not native_dir.is_dir() or not any(native_dir.glob("*.json")):
            _rebuild_native_from_rollouts(job_dir, domain)
        if not native_dir.is_dir():
            warn(TAG, f"no native/ after rebuild: {job_dir}")
            continue
        for native_path in sorted(native_dir.glob("*.json")):
            jobs.append(
                NativeJob(
                    domain=domain,
                    job_dir=job_dir,
                    native_path=native_path,
                    file_name=_file_name(rollouts_root, job_dir, native_path.stem),
                )
            )
    return jobs


def stage_job(job: NativeJob, repo: Path, questions: dict[str, str] | None) -> Path:
    native = json.loads(job.native_path.read_text(encoding="utf-8"))
    if job.domain == "math":
        if not questions:
            raise ValueError("math staging requires annotation questions")
        payload = to_paper_math(native, questions)
        ann = MATH_ANN
    else:
        payload = to_paper_writing(native)
        ann = WRITE_ANN
    dest = repo / "simulation" / "output" / ann / f"{job.file_name}.json"
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(json.dumps(payload), encoding="utf-8")
    return dest


def _run_py(script: Path, args: list[str], *, cwd: Path) -> int:
    cmd = [sys.executable, str(script), *args]
    log(TAG, " ".join(cmd))
    proc = subprocess.run(cmd, cwd=str(cwd), check=False)
    return proc.returncode


def _process_batch(process_py: Path, batch_file: Path, *, model: str, poll: int, cwd: Path) -> int:
    if not batch_file.is_file():
        log(TAG, f"no batch file {batch_file} (nothing new to score)")
        return 0
    return _run_py(
        process_py,
        [
            "--batch_file",
            str(batch_file),
            "--model",
            model,
            "--poll_interval",
            str(poll),
        ],
        cwd=cwd,
    )


def _math_done(eval_root: Path, file_name: str) -> tuple[bool, bool]:
    answers = eval_root / "evaluation_outputs" / "extracted_answer" / f"{file_name}.json"
    if not answers.is_file():
        return False, False
    data = json.loads(answers.read_text(encoding="utf-8"))
    nested = data.get("answers") or data
    has_correct = False

    def walk(node: Any) -> None:
        nonlocal has_correct
        if isinstance(node, dict):
            if "correctness" in node:
                has_correct = True
                return
            for v in node.values():
                walk(v)

    walk(nested)
    return True, has_correct


def _writing_done(eval_root: Path, file_name: str, kind: str) -> bool:
    return (eval_root / "evaluation_outputs" / kind / f"{file_name}.json").is_file()


def score_math(
    job: NativeJob,
    repo: Path,
    *,
    model: str,
    poll: int,
    skip_completed: bool,
) -> None:
    eval_root = repo / "evaluation" / "math_tutoring"
    scripts = eval_root / "scripts"
    extracted, correct = _math_done(eval_root, job.file_name)
    if not (skip_completed and extracted):
        rc = _run_py(
            scripts / "generate_batch_prompts_for_answer_extraction.py",
            ["--file_name", job.file_name, "--annotation_id", MATH_ANN, "--terminate_help", "false"],
            cwd=scripts,
        )
        if rc != 0:
            raise RuntimeError(f"math extract prompt gen failed ({rc}) for {job.file_name}")
        rc = _process_batch(
            scripts / "process_batch_evaluation.py",
            eval_root / "batch_prompts" / "extracted_answer" / f"{job.file_name}.jsonl",
            model=model,
            poll=poll,
            cwd=scripts,
        )
        if rc != 0:
            raise RuntimeError(f"math extract batch failed ({rc}) for {job.file_name}")
    else:
        log(TAG, f"skip math extract (exists): {job.file_name}")

    extracted, correct = _math_done(eval_root, job.file_name)
    if skip_completed and correct:
        log(TAG, f"skip math correctness (exists): {job.file_name}")
        return
    if not extracted:
        warn(TAG, f"no extracted answers; skip correctness for {job.file_name}")
        return
    rc = _run_py(
        scripts / "generate_batch_prompts_for_correctness_check.py",
        ["--file_name", job.file_name, "--annotation_id", MATH_ANN],
        cwd=scripts,
    )
    if rc != 0:
        raise RuntimeError(f"math correctness prompt gen failed ({rc}) for {job.file_name}")
    rc = _process_batch(
        scripts / "process_batch_evaluation.py",
        eval_root / "batch_prompts" / "extracted_answer" / f"{job.file_name}_correctness.jsonl",
        model=model,
        poll=poll,
        cwd=scripts,
    )
    if rc != 0:
        raise RuntimeError(f"math correctness batch failed ({rc}) for {job.file_name}")


def score_writing(
    job: NativeJob,
    repo: Path,
    *,
    model: str,
    poll: int,
    skip_completed: bool,
) -> None:
    eval_root = repo / "evaluation" / "document_creation"
    scripts = eval_root / "scripts"
    steps: list[tuple[str, list[str], Path, str]] = [
        (
            "generate_batch_prompts_for_document_extraction.py",
            [
                "--file_name",
                job.file_name,
                "--annotation_id",
                WRITE_ANN,
                "--terminate_help",
                "false",
                "--evaluator_model",
                model,
            ],
            eval_root / "batch_prompts" / "extracted_document" / f"{job.file_name}.jsonl",
            "extracted_document",
        ),
        (
            "generate_batch_prompts_for_rating.py",
            [
                "--file_name",
                job.file_name,
                "--annotation_id",
                WRITE_ANN,
                "--aspect",
                "document",
                "--terminate_help",
                "false",
                "--evaluator_model",
                model,
            ],
            eval_root / "batch_prompts" / "document_rating" / f"{job.file_name}.jsonl",
            "document_rating",
        ),
        (
            "generate_batch_prompts_for_rating.py",
            [
                "--file_name",
                job.file_name,
                "--annotation_id",
                WRITE_ANN,
                "--aspect",
                "interaction",
                "--terminate_help",
                "false",
                "--evaluator_model",
                model,
            ],
            eval_root / "batch_prompts" / "interaction_rating" / f"{job.file_name}.jsonl",
            "interaction_rating",
        ),
    ]
    for gen_name, gen_args, batch_file, done_kind in steps:
        if skip_completed and _writing_done(eval_root, job.file_name, done_kind):
            log(TAG, f"skip writing {done_kind} (exists): {job.file_name}")
            continue
        rc = _run_py(scripts / gen_name, gen_args, cwd=scripts)
        if rc != 0:
            raise RuntimeError(f"writing {done_kind} prompt gen failed ({rc}) for {job.file_name}")
        rc = _process_batch(
            scripts / "process_batch_evaluation.py",
            batch_file,
            model=model,
            poll=poll,
            cwd=scripts,
        )
        if rc != 0:
            raise RuntimeError(f"writing {done_kind} batch failed ({rc}) for {job.file_name}")


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--rollouts-root",
        type=Path,
        default=storage_root() / "outputs" / "rollouts",
        help="Directory to walk for rollout.simulatorarena.jsonl",
    )
    parser.add_argument(
        "--domains",
        default="math,writing",
        help="Comma-separated: math, writing",
    )
    parser.add_argument("--evaluator-model", default="gpt-5-mini")
    parser.add_argument("--poll-interval", type=int, default=30)
    parser.add_argument("--no-skip", action="store_true", help="Re-run steps even if outputs exist")
    parser.add_argument("--dry-run", action="store_true", help="List jobs; do not call OpenAI")
    parser.add_argument(
        "--limit",
        type=int,
        default=0,
        help="Score at most N native files (0 = all)",
    )
    args = parser.parse_args(argv)

    domains = {d.strip() for d in args.domains.split(",") if d.strip()}
    root = Path(args.rollouts_root)
    if not root.is_dir():
        raise SystemExit(f"--rollouts-root is not a directory: {root}")

    repo = repo_path()
    jobs = discover_jobs(root, domains=domains)
    if args.limit:
        jobs = jobs[: args.limit]
    log(TAG, f"found {len(jobs)} native file(s) under {root}")
    for job in jobs:
        log(TAG, f"  [{job.domain}] {job.file_name}  <- {job.native_path}")
    if args.dry_run or not jobs:
        return

    if not os.environ.get("OPENAI_API_KEY"):
        raise SystemExit("OPENAI_API_KEY is required (OpenAI batch API)")

    questions = load_math_questions(repo) if any(j.domain == "math" for j in jobs) else None
    skip = not args.no_skip
    failures: list[str] = []
    for job in tqdm(jobs, desc="simarena-score", unit="job"):
        try:
            stage_job(job, repo, questions)
            if job.domain == "math":
                score_math(
                    job,
                    repo,
                    model=args.evaluator_model,
                    poll=args.poll_interval,
                    skip_completed=skip,
                )
            else:
                score_writing(
                    job,
                    repo,
                    model=args.evaluator_model,
                    poll=args.poll_interval,
                    skip_completed=skip,
                )
        except Exception as exc:  # noqa: BLE001
            warn(TAG, f"{job.file_name}: {exc}")
            failures.append(f"{job.file_name}: {exc}")
    if failures:
        raise SystemExit(f"{len(failures)} job(s) failed:\n" + "\n".join(failures))
    log(TAG, "done")


if __name__ == "__main__":
    main()
