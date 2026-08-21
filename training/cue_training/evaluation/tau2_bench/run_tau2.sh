#!/bin/bash
set -euo pipefail

unset VIRTUAL_ENV
unset UV_PROJECT_ENVIRONMENT
REPO_ROOT=$(cd "$(dirname "$0")/../../../.." && pwd)
cd "${REPO_ROOT}"

# tau2-bench task-completion eval with CUE/baseline user simulators (customer-service).
#
# Prereqs (run once):
#   clone tau2-bench under training/cue_training/evaluation/tau2_bench/external/
#   uv pip install -e training/cue_training/evaluation/tau2_bench/external/tau2-bench
#   export OPENAI_API_KEY=...  (agent + API user sims + decoder sim)
#
# EMBEDDINGS must be the CUE encoder export over the tau_usi eval set, produced by the
# same encoder checkpoint the decoder was trained on.

STORAGE_ROOT=${CUE_STORAGE_ROOT:-${REPO_ROOT}/training/artifacts}
DATA_DIR=${DATA_DIR:-${STORAGE_ROOT}/data}
MODEL_DIR=${MODEL_DIR:-${STORAGE_ROOT}/models}
TAG=${TAG:-discovered_e5-base-v2}
OUT_DIR=${OUT_DIR:-${STORAGE_ROOT}/outputs/tau2}
CASES_DIR=${CASES_DIR:-${OUT_DIR}/cases}

NORMALIZED=${NORMALIZED:-${DATA_DIR}/mirrorbench/data/tau_usi/normalized.jsonl}
EMBEDDINGS=${EMBEDDINGS:-${STORAGE_ROOT}/embeddings/cue/tau2.json}
DECODER_DIR=${DECODER_DIR:-${MODEL_DIR}/${TAG}/decoder_qwen3_1.7b}

# Default variants exclude the expensive/GPU-heavy decoder and base_local; add them
# explicitly via VARIANTS or INCLUDE_DECODER=1 / INCLUDE_BASE_LOCAL=1.
VARIANTS=${VARIANTS:-"baseline:realusersim baseline:usp baseline:ppol baseline:userlm base_api"}
if [ "${INCLUDE_DECODER:-0}" = "1" ]; then VARIANTS="${VARIANTS} decoder"; fi
if [ "${INCLUDE_BASE_LOCAL:-0}" = "1" ]; then VARIANTS="${VARIANTS} base_local"; fi
ARMS=${ARMS:-"paired sample_shuffled dataset_mean"}
AGENT_LLM=${AGENT_LLM:-gpt-5.2}
SIM_MODEL=${SIM_MODEL:-gpt-5.4-mini}
CONCURRENCY=${CONCURRENCY:-1}
TRIALS=${TRIALS:-1}
DEVICE=${DEVICE:-cuda}
MAX_EPISODES_FLAG=()
if [ -n "${MAX_EPISODES:-}" ]; then MAX_EPISODES_FLAG=(--max_episodes "${MAX_EPISODES}"); fi

echo "=== [prep] building tau2 cases from ${NORMALIZED} + ${EMBEDDINGS} ==="
uv run --group research cue-tau2-eval prep \
  --normalized "${NORMALIZED}" \
  --embeddings "${EMBEDDINGS}" \
  --out_dir "${CASES_DIR}" \
  --arms ${ARMS}

echo "=== [run] tau2 rollouts for variants: ${VARIANTS} ==="
uv run --group research cue-tau2-eval run \
  --cases_dir "${CASES_DIR}" \
  --out_dir "${OUT_DIR}" \
  --variants ${VARIANTS} \
  --arms ${ARMS} \
  --agent_llm "${AGENT_LLM}" \
  --sim_model "${SIM_MODEL}" \
  --decoder_dir "${DECODER_DIR}" \
  --concurrency "${CONCURRENCY}" \
  --trials "${TRIALS}" \
  --device "${DEVICE}" \
  ${MAX_EPISODES_FLAG[@]+"${MAX_EPISODES_FLAG[@]}"}

echo "Done. Summary: ${OUT_DIR}/summary.json ; MirrorBench rollouts: ${OUT_DIR}/rollout.tau2.jsonl"
