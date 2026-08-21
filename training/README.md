# CUE training and evaluation

Checkout-only research stack for CUE. Relicensed Apache-2.0 with the rest of
this repository. It is **not** part of the published `cue-simulator` wheel.

## Install

From the repository root (Python 3.12 or 3.13):

```bash
uv sync --group research
uv run --group research cue-train --help
uv run --group research cue-data --help
uv run --group research cue-rollouts --help
uv run --group research cue-metrics --help
```

Tau2 failure-mode analysis has its own optional dependencies:

```bash
uv sync --group fma
uv run --group fma cue-fma --help
```

Always pass `--group research` on later `uv run` commands. A plain `uv run`
re-syncs default groups and can uninstall this package. On a 3.14 interpreter:

```bash
uv run --python 3.13 --group research cue-train --help
```

Do not use `--only-group research`; that omits the root `cue-simulator` project.

vLLM serving stays in a separate Python 3.12 environment
(`uv pip install --project training --python 3.12 --group vllm`).

## Commands

| CLI | purpose |
|-----|---------|
| `cue-train` | Joint train, refine, encode, export |
| `cue-data` | Pull annotations, preprocess corpora |
| `cue-rollouts` | Generate environment rollouts |
| `cue-metrics` | Score the eight paper metrics |
| `cue-baseline` | Train baseline simulators |
| `cue-tau2-eval` / `cue-simarena-eval` | Environment harnesses |
| `cue-simarena-score` | Score SimulatorArena rollouts |
| `cue-fma` | Run Tau2 failure-mode analysis |

Tau2 and SimulatorArena external trees are not shipped. Clone them under
`cue_training/evaluation/tau2_bench/external/tau2-bench` and
`cue_training/evaluation/simulatorarena/external/SimulatorArena` if you need
those harnesses.

## Metrics

`cue-metrics` exposes exactly:

- `classifier/sim2real` (Nat-S2R)
- `judge/turing_sonnet_qwen` (Nat-TT)
- `mimicry/wegmann_ava` (Mim-AVA)
- `mimicry/paired_audit` (Mim-PT3)
- `coverage/sim2real_behavioral` (Cov-S2RChamfer)
- `coverage/styledistance_behavioral` (Cov-SDChamfer)
- `env/tau2_success_rate`
- `env/tau2_task_success` (pairwise success F1)

## Failure-mode analysis

`cue-fma` implements the Appendix D human-review workflow: ingest failed human
and simulator trajectories, sample and propose labels, review three batches in
the local UI, tag the remainder, and compute failure-mix TVD. Source, prompts,
UI assets, and plotting code are versioned under `cue_training/evaluation/fma`;
generated artifacts default to the gitignored `training/artifacts/fma/` directory
and are not included in this repository.

## Tests

```bash
uv run --group research --group fma pytest training/tests
```
