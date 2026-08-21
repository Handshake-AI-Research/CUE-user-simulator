#!/bin/bash
set -euo pipefail

unset VIRTUAL_ENV
unset UV_PROJECT_ENVIRONMENT
REPO_ROOT=$(cd "$(dirname "$0")/../../../.." && pwd)
cd "${REPO_ROOT}"

# SimulatorArena document-creation proxy eval with CUE/baseline user simulators.
#
# Prereqs (run once):
#   clone SimulatorArena under training/cue_training/evaluation/simulatorarena/external/
#   (SimulatorArena deps: prefer sys.path + vendored deps, or a separate venv; set
#    SIMULATORARENA_PATH if the submodule lives elsewhere)
#   export OPENAI_API_KEY=...  (assistant + API user sims)
#
# EMBEDDINGS must be the CUE encoder export over the writing (simulatorarena) eval set,
# from the same encoder checkpoint the decoder was trained on.

STORAGE_ROOT=${CUE_STORAGE_ROOT:-${REPO_ROOT}/training/artifacts}
DATA_DIR=${DATA_DIR:-${STORAGE_ROOT}/data}
MODEL_DIR=${MODEL_DIR:-${STORAGE_ROOT}/models}
TAG=${TAG:-discovered_e5-base-v2}
OUT_DIR=${OUT_DIR:-${STORAGE_ROOT}/outputs/simulatorarena}

SIMARENA=${SIMARENA:-training/cue_training/evaluation/simulatorarena/external/SimulatorArena}
ANNOTATIONS=${ANNOTATIONS:-${SIMARENA}/data/document_creation_annotations.json}
EMBEDDINGS=${EMBEDDINGS:-${STORAGE_ROOT}/embeddings/cue/simarena_writing.json}
BACKGROUND=${BACKGROUND:-${SIMARENA}/data/document_creation_user_simulator_background.json}
PROFILES_DIR=${PROFILES_DIR:-${SIMARENA}/data/user_simulator_profiles/document_creation}
DECODER_DIR=${DECODER_DIR:-${MODEL_DIR}/${TAG}/decoder_qwen3_1.7b}

VARIANTS=${VARIANTS:-"baseline:realusersim baseline:usp baseline:ppol baseline:userlm base_api"}
if [ "${INCLUDE_DECODER:-0}" = "1" ]; then VARIANTS="${VARIANTS} decoder"; fi
if [ "${INCLUDE_BASE_LOCAL:-0}" = "1" ]; then VARIANTS="${VARIANTS} base_local"; fi
ARMS=${ARMS:-"paired sample_shuffled dataset_mean"}
ASSISTANT_MODEL=${ASSISTANT_MODEL:-gpt-5.2}
SIM_MODEL=${SIM_MODEL:-gpt-5.4-mini}
MAX_TURNS=${MAX_TURNS:-12}
DEVICE=${DEVICE:-cuda}
MAXCONV_FLAG=()
if [ -n "${MAX_CONVERSATIONS:-}" ]; then MAXCONV_FLAG=(--max_conversations "${MAX_CONVERSATIONS}"); fi

uv run --group research cue-simarena-eval run \
  --annotations "${ANNOTATIONS}" \
  --embeddings "${EMBEDDINGS}" \
  --out_dir "${OUT_DIR}" \
  --background "${BACKGROUND}" \
  --profiles_dir "${PROFILES_DIR}" \
  --variants ${VARIANTS} \
  --arms ${ARMS} \
  --assistant_model "${ASSISTANT_MODEL}" \
  --sim_model "${SIM_MODEL}" \
  --decoder_dir "${DECODER_DIR}" \
  --max_turns "${MAX_TURNS}" \
  --device "${DEVICE}" \
  ${MAXCONV_FLAG[@]+"${MAXCONV_FLAG[@]}"}

echo "Done. Summary: ${OUT_DIR}/summary.json ; MirrorBench rollouts: ${OUT_DIR}/rollout.simulatorarena.jsonl"
echo "SimulatorArena-format conversations under ${OUT_DIR}/native/ (reconcile schema + run its terminate/eval for correlation)."
