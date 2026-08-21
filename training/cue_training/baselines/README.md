# User-simulator baselines

Four user-simulator baselines that each (a) **train** when needed and (b) **rollout**
a MirrorBench-compatible JSONL for the three evaluation domains (coding,
customer service, writing). All baselines share one closed-loop rollout harness
and a single CLI.

| Baseline      | Paper                              | Training                                                        | Simulator |
|---------------|------------------------------------|----------------------------------------------------------------|-----------|
| `userlm`      | UserLM (arXiv:2510.06552)          | none (pretrained `microsoft/UserLM-8b`)                        | HF UserLM-8b with generation guardrails |
| `realusersim` | RealUserSim (arXiv:2605.20204)     | WildChat-4.8M filter → per-user GPT-4o Executable Persona Manuals (~7.3k) | LLM grounded on a randomly sampled WildChat persona (`as_is`) |
| `ppol`        | PPol (arXiv:2605.12894)            | OpenEvolve program search over G(c,D,N), vendored upstream (`baselines/ppol/external/persona-policies`); run via `scripts/train_ppol.sh` | base LLM + task-conditioned evolved persona (per-task, injected by the tau2 sidecar) |
| `usp`         | USP (arXiv:2502.18968)             | none (published `wangkevin02/USP`)                             | vLLM chat-completions with official chat template + implicit profile; `sample_diverse` uses `wangkevin02/LMSYS-USP` |

### RealUserSim (paper WildChat pool)

Training defaults to importing the authors' released profiles
(`Salesforce/RealUserSim`, ~7,273 users) verbatim, then eval `as_is` randomly
samples one per task (τ-bench “Real Persona” setup).

```bash
# Default: use the published profiles exactly:
uv run --group research cue-rollouts baseline realusersim --train
# Rebuild from scratch instead (WildChat-4.8M filter + GPT-4o × ~7k users):
REALUSERSIM_REBUILD=1 uv run --group research cue-rollouts baseline realusersim --train
```

The rebuild path matches Appendix A: stream `allenai/WildChat-4.8M` (3.2M
non-toxic), keep English multi-turn GPT-4o conversations with ≥3 substantive
turns after trimming greeting/thank-you bookends (~21,637 trajs / ~7,311 users),
then GPT-4o extracts one Executable Persona Manual **per user**.

### PPol (faithful reimplementation)

`ppol` wraps the upstream Persona-Policies repo (clone it under
`cue_training/baselines/ppol/external/persona-policies`). Its personas are produced by the
OpenEvolve-evolved generator `G(c, D, N)`, not by our old genome approximation
(now retired). Training is heavy and tau2-coupled; the unified rollouts CLI
invokes paper-parity training (wrapping `scripts/train_ppol.sh`) when `--train`
is set and the freeze is missing:

```bash
# Paper-parity train + rollout (Tau2/SimArena/PRISM from training/configs/rollouts.json):
uv run --group research cue-rollouts baseline ppol --train
# Train all four rollout sims in parallel (default standalone):
./training/scripts/train_ppol.sh
# Train one sim only:
PPOL_SINGLE=1 PPOL_DOMAIN=airline PPOL_SIM_MODEL=gpt-5.4-mini ./training/scripts/train_ppol.sh
# Force re-evolve through rollouts:
uv run --group research cue-rollouts baseline ppol --force-train
```

Train and rollout share the same user system-prompt structure: tau2 native guidelines +
`<scenario>`, then the CUE `Task description:` / `Domain or topic:` block, then the
upstream `PERSONA_INJECTION_TEMPLATE` wrapping the evolved persona.

PPol is an `as_is` baseline with **no** `paired`/`sample_shuffled` arms: its personas
are conditioned on the task scenario and can contain task facts, so cross-task
shuffling is unsafe, and a fingerprint-matched "paired" persona would only be
determinable post-hoc (a leaky comparison). The human reference for the
discriminator/coverage is the shipped `tau_bench_human.json`, which is the same
tau-usi source we evaluate on. Persona generation uses paper ``G(c, D, N)`` with
``format_user_scenario_c`` for ``c``, final ``N=10``, and ``PPOL_GEN_MODEL``
(default Gemini Flash via OpenRouter), decoupled from the rollout ``SIM_MODEL``.
SimArena/PRISM reuse the τ²-generated persona pool (no domain-native G(c) source).

## Usage

```bash
# Low-level train (no-op for userlm; data prep / fine-tuning for the others)
uv run --group research cue-baseline train realusersim
# USP uses the published HF checkpoint (wangkevin02/USP); train is a no-op record.

# Canonical env rollouts (Tau2 / SimulatorArena / PRISM) via the unified endpoint:
uv run --group research cue-rollouts baseline base realusersim usp
uv run --group research cue-rollouts baseline usp ppol --train
uv run --group research cue-rollouts baseline all --dry-run
```

Paper-parity `--train` uses `baselines.common.manifest.train_paper` (fail-closed
artifact manifests). For USP that only records `wangkevin02/USP` — there is no
local SFT/RLCC. Outputs land under
`$CUE_STORAGE_ROOT/outputs/rollouts/{run_id}/{benchmark}/{domain}/baseline/{method}/{simulator}/`.

### Key flags (`uv run --group research cue-rollouts ...`)

- `--config` — shared `training/configs/rollouts.json` (benchmarks, simulators, assistant, resources).
- `--train` / `--force-train` — paper-parity baseline training before rollout (USP: HF record only).
- `--dry-run` — print the resolved job DAG / GPU-port plan without side effects.
- `--resume`, `--limit`, `--run-id`.

Low-level `baseline train` still accepts `--data-path`, API knobs, and prep flags for
baselines that need data prep (e.g. realusersim).

## MirrorBench wiring

Each rollout file is consumed by MirrorBench's existing replay path:

```yaml
dataset:
  kind: jsonl/rollout
  path: $CUE_STORAGE_ROOT/baselines/<baseline>/rollout.<domain>.jsonl
task:
  kind: rollout/replay
adapter:
  kind: static/replay
```

MirrorBench loads `rollout_conversation` into the scored artifact and compares
the synthetic user turns against `real_conversation`. The record schema is:

```json
{"episode_id": "...::<baseline>", "task_id": "...",
 "real_conversation": [...],
 "rollout_conversation": [{"role": "user", "content": "..."}, {"role": "assistant", "content": "..."}],
 "metadata": {"rollout_mode": "<baseline>", "domain": "...", ...}}
```
