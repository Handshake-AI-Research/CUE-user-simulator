# Failure Mode Analysis (Appendix D)

Thin CUE tool for coding tau2 agent failures: sample 100 failed trajectories,
LLM-propose a mode with cited turns + explanation, human-review in a local UI, repeat
for 3 batches, then tag the remainder and compute failure-mix TVD.

## Setup

```bash
uv sync --group fma   # fastapi, uvicorn, jinja2, matplotlib
export CUE_STORAGE_ROOT=...
export OPENAI_API_KEY=...   # or ANTHROPIC_API_KEY for claude-fable-5
# optional: export FMA_MODEL=claude-fable-5
```

## Protocol → commands

| Appendix D step | Command |
|-----------------|---------|
| Collect failed human + sim trajs | `uv run --group fma cue-fma ingest ...` |
| Batch of 100 + LLM propose | `uv run --group fma cue-fma batch --run $RUN --n 100` |
| Human audit | `uv run --group fma cue-fma review --run $RUN --batch 1` |
| Optional taxonomy merge | `uv run --group fma cue-fma discover --run $RUN --batch 1` |
| Repeat batches 2–3 | `batch` / `review` again |
| Tag remainder | `uv run --group fma cue-fma tag --run $RUN` |
| Sim error % + attribution / agent TVD | `uv run --group fma cue-fma tvd --run $RUN` |
| Fidelity × calibration Pearson table | `uv run --group fma cue-fma compare` |

Example:

```bash
RUN=tau2-baselines-v1

uv run --group fma cue-fma ingest --out "$RUN" \
  --human data/evaluation/tau_usi/normalized.jsonl \
  --sim '$CUE_STORAGE_ROOT/outputs/rollouts/seed-*/**/rollout.tau2.jsonl' \
  --arms paired as_is

uv run --group fma cue-fma batch --run "$RUN" --n 100 --seed 0
uv run --group fma cue-fma review --run "$RUN" --batch 1   # http://127.0.0.1:8765
# In the review UI taxonomy bar: Add / Rename / Merge / Edit description anytime.
# Rename+merge also rewrite labels in proposals/decisions/tagged for this run.

# after 3 audited batches:
uv run --group fma cue-fma tag --run "$RUN"
uv run --group fma cue-fma tvd --run "$RUN"

# After metrics leaderboard + distributions.json exist:
uv run --group fma cue-fma compare \
  --leaderboard output/metrics/tau2_customer-service/leaderboard.json \
  --distributions fma/distributions.json \
  --out-dir fma/figures
# Optional: --plot for a 6×3 scatter grid
```

`--out` / `--run` may be a path or a run id under `$CUE_STORAGE_ROOT/fma/<id>/`.

Each sim gets `source_id = <method>-<simulator>|<arm>` (e.g. `ppol-llama|as_is`),
read from the job path (`.../baseline/ppol/llama/` or `.../cue/general/llama/`) rather
than the filename, which is `rollout.tau2.jsonl` for every model. Multiseed wrappers
(`seed-N/`) are omitted from `source_id` so TVD pools **raw failure counts across seeds**
under one distribution per method/sim/arm; `primary_key` still includes `seed-N` so the
same episode from different seeds stays distinct for tagging/review. That is the key TVD
reports by, so globs over many models and seeds are safe.

## Artifacts

```
$CUE_STORAGE_ROOT/fma/<run_id>/
  corpus.jsonl
  coverage.json
  taxonomy.json          # [{name, description, examples:[...]}]
  batches/01/{sample,proposals,decisions}.jsonl
  tagged.jsonl
  distributions.json     # TVD vs human
```

## Export expectations

Prefer new tau2 MirrorBench rows with `metadata.full_conversation` (tool calls + results)
and `metadata.reward_report` (failed checks only). Metrics still use stripped dialogue +
scalar reward. Legacy dialogue-only JSONLs still ingest; coding degrades gracefully.
Human τ-USI inline `<function=…>` / `<|tool|>` markup is expanded into the same turn schema.
