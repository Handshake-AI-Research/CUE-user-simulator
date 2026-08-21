#!/bin/bash
set -euo pipefail

# Start a vLLM OpenAI-compatible server for the local user-sim model (e.g. base_local
# Llama). Self-contained: creates a dedicated Python 3.12 venv (vLLM has no 3.13 wheel and
# CUE's .venv is 3.13), installs vLLM + Ray (vLLM's multi-GPU backend), and launches
# `vllm serve`, then blocks until it's ready. Point the CUE sidecar at it:
#     ... -m cue_training.evaluation.common.sidecar --vllm_base_url http://127.0.0.1:${PORT}/v1
#
# GPU node only. HF-gated models (Llama) need a token: export HF_TOKEN=... first.
#
# Env knobs (all optional):
#   MODEL           model to serve (default meta-llama/Llama-3.1-8B-Instruct)
#   PORT            server port (default 8000)
#   GPUS            CUDA_VISIBLE_DEVICES for this server (default 0; e.g. "0,1" for TP=2)
#   TP              tensor-parallel size = #GPUs for this server (default 1)
#   GPU_MEM_UTIL    vLLM --gpu-memory-utilization (default 0.90)
#   MAX_MODEL_LEN   cap context length (default: model default)
#   VLLM_VENV       venv path (default <repo>/.venv-vllm)
#   PYVER           Python for the vLLM venv (default 3.12)
#   INSTALL         1 to (re)install vLLM+Ray, 0 to skip if already installed (default 1)
#   WAIT_TIMEOUT    seconds to wait for readiness (default 1800)

unset VIRTUAL_ENV
unset UV_PROJECT_ENVIRONMENT

CUE_ROOT=$(cd "$(dirname "$0")/.." && pwd)
cd "${CUE_ROOT}"

MODEL=${MODEL:-meta-llama/Llama-3.1-8B-Instruct}
PORT=${PORT:-8000}
HOST=${HOST:-0.0.0.0}
GPUS=${GPUS:-0}
TP=${TP:-1}
GPU_MEM_UTIL=${GPU_MEM_UTIL:-0.92}
# Cap context to maximize KV-cache concurrency (user-sim turns are short; 16k is ample for
# tau2/customer-service). Bump for very long instructions. MAX_NUM_SEQS raises the vLLM
# batch ceiling to soak up high client concurrency on an 80GB GPU.
MAX_MODEL_LEN=${MAX_MODEL_LEN:-16384}
MAX_NUM_SEQS=${MAX_NUM_SEQS:-512}
PYVER=${PYVER:-3.12}
INSTALL=${INSTALL:-1}
WAIT_TIMEOUT=${WAIT_TIMEOUT:-1800}

# Seed workers export CUDA_VISIBLE_DEVICES=<physical> and pass GPUS=0 (logical index into
# that mask). Remap so we do not clobber the parent pin and land every seed on GPU 0.
PARENT_CVD="${CUDA_VISIBLE_DEVICES:-}"
if [ -n "${PARENT_CVD}" ]; then
  IFS=',' read -r -a _mask <<< "${PARENT_CVD}"
  IFS=',' read -r -a _idx <<< "${GPUS}"
  _mapped=()
  for _i in "${_idx[@]}"; do
    _i="$(printf '%s' "${_i}" | tr -d '[:space:]')"
    if [ -z "${_i}" ]; then
      continue
    fi
    if [ "${_i}" -lt 0 ] 2>/dev/null || [ "${_i}" -ge "${#_mask[@]}" ] 2>/dev/null; then
      echo "ERROR: GPUS index ${_i} out of range for CUDA_VISIBLE_DEVICES=${PARENT_CVD}" >&2
      exit 1
    fi
    _mapped+=("${_mask[_i]}")
  done
  [ "${#_mapped[@]}" -gt 0 ] || { echo "ERROR: empty GPUS after remapping through CUDA_VISIBLE_DEVICES=${PARENT_CVD}" >&2; exit 1; }
  GPUS="$(IFS=,; echo "${_mapped[*]}")"
fi

# The venv (vLLM+torch+CUDA, several GB), the model download (~16GB), and the uv/vLLM
# build caches must NOT land on the small Anyscale working-dir volume. Route them to a
# big disk: node-local /mnt/local_storage (fast), else shared /mnt/cluster_storage, else
# the repo. Override CACHE_ROOT (or any of the individual dirs) as needed.
if [ -z "${CACHE_ROOT:-}" ]; then
  if [ -d /mnt/local_storage ]; then CACHE_ROOT=/mnt/local_storage/cue_vllm
  elif [ -d /mnt/cluster_storage ]; then CACHE_ROOT=/mnt/cluster_storage/cue_vllm
  else CACHE_ROOT="${CUE_ROOT}"; fi
fi
mkdir -p "${CACHE_ROOT}"
VLLM_VENV=${VLLM_VENV:-${CACHE_ROOT}/.venv-vllm}
export HF_HOME=${HF_HOME:-${CACHE_ROOT}/hf}
export HUGGINGFACE_HUB_CACHE=${HUGGINGFACE_HUB_CACHE:-${HF_HOME}/hub}
export UV_CACHE_DIR=${UV_CACHE_DIR:-${CACHE_ROOT}/uv}
export VLLM_CACHE_ROOT=${VLLM_CACHE_ROOT:-${CACHE_ROOT}/vllm}
export XDG_CACHE_HOME=${XDG_CACHE_HOME:-${CACHE_ROOT}/xdg}
mkdir -p "${HF_HOME}" "${UV_CACHE_DIR}" "${VLLM_CACHE_ROOT}" "${XDG_CACHE_HOME}"
echo "=== caches -> ${CACHE_ROOT} (HF_HOME=${HF_HOME}, venv=${VLLM_VENV}) ==="

if [ "${INSTALL}" = "1" ]; then
  echo "=== installing vLLM + Ray (+ ninja) into ${VLLM_VENV} (Python ${PYVER}) ==="
  # Create the venv only if missing (uv venv errors on an existing one); the pip install is
  # idempotent, so a second server sharing VLLM_VENV -- or a re-run -- reuses it. VLLM_VENV_CLEAR=1
  # forces a fresh venv.
  if [ "${VLLM_VENV_CLEAR:-0}" = "1" ]; then
    uv venv --clear --python "${PYVER}" "${VLLM_VENV}"
  elif [ ! -d "${VLLM_VENV}" ]; then
    uv venv --python "${PYVER}" "${VLLM_VENV}"
  fi
  # Install the pinned GPU serving stack from the pyproject `vllm` dependency group so
  # versions stay reproducible/isolated from the CUE 3.13 env. Fall back to explicit
  # packages if this uv predates `uv pip install --group`. (ninja: needed if any
  # FlashInfer/torch kernel JIT-compiles at startup.)
  uv pip install --python "${VLLM_VENV}/bin/python" --group vllm --project "${CUE_ROOT}" \
    || uv pip install --python "${VLLM_VENV}/bin/python" vllm "ray[default]" ninja
fi

# FlashInfer's sampler JIT-compiles a CUDA kernel at startup (needs ninja + nvcc, often
# absent on serving nodes -> "FileNotFoundError: 'ninja'"). Disable it by default so vLLM
# uses the native sampler (no build step); negligible cost for short user-sim generations.
# Set VLLM_USE_FLASHINFER_SAMPLER=1 to re-enable (requires ninja + CUDA toolkit).
export VLLM_USE_FLASHINFER_SAMPLER=${VLLM_USE_FLASHINFER_SAMPLER:-0}
# Anyscale workspaces already run Ray; vLLM V1's multiprocess engine-core often dies with
# empty "Failed core proc(s): {}" (OOM-kill or Ray-in-Ray). Prefer the V0 engine unless
# the caller opts in. spawn avoids CUDA fork issues after the parent has touched the GPU.
export VLLM_USE_V1=${VLLM_USE_V1:-0}
export VLLM_WORKER_MULTIPROC_METHOD=${VLLM_WORKER_MULTIPROC_METHOD:-spawn}

VLLM_BIN="${VLLM_VENV}/bin/vllm"
if [ ! -x "${VLLM_BIN}" ]; then
  echo "ERROR: vllm not found at ${VLLM_BIN}. Re-run with INSTALL=1." >&2
  exit 1
fi

if [ -z "${HF_TOKEN:-}${HUGGING_FACE_HUB_TOKEN:-}" ]; then
  echo "WARNING: no HF_TOKEN/HUGGING_FACE_HUB_TOKEN set; gated models (e.g. Llama) will fail to download." >&2
fi

# Local checkpoint dirs must exist and look like a HF model (config.json). A missing dir is
# misread as a Hub repo id and fails with a cryptic HFValidationError.
case "${MODEL}" in
  /*|./*)
    if [ ! -f "${MODEL}/config.json" ]; then
      echo "ERROR: local MODEL ${MODEL} is missing or incomplete (no config.json)." >&2
      echo "  For usp/turing_rl, run merge first, e.g.:" >&2
      echo "    python -m cue_training.baselines.turing_rl.merge --turing_dir <artifacts>/turing_rl" >&2
      exit 1
    fi
    ;;
esac

ARGS=(serve "${MODEL}" --host "${HOST}" --port "${PORT}"
      --gpu-memory-utilization "${GPU_MEM_UTIL}" --tensor-parallel-size "${TP}")
# Alias the served model name (clients use this instead of the model path). Handy when
# MODEL is a local dir, e.g. SERVED_MODEL_NAME=usp-merged for the merged USP policy.
if [ -n "${SERVED_MODEL_NAME:-}" ]; then
  ARGS+=(--served-model-name "${SERVED_MODEL_NAME}")
fi
if [ -n "${MAX_MODEL_LEN}" ]; then
  ARGS+=(--max-model-len "${MAX_MODEL_LEN}")
fi
if [ -n "${MAX_NUM_SEQS}" ]; then
  ARGS+=(--max-num-seqs "${MAX_NUM_SEQS}")
fi
# Convert a generative decoder into a pooling/embedding server (CONVERT=embed -> LAST pooling +
# normalize, exposes /v1/embeddings; used for the OSS hidden-state probe features). Disables
# generation endpoints, so run a separate generation server for text generation.
if [ -n "${CONVERT:-}" ]; then
  ARGS+=(--convert "${CONVERT}")
fi
# Enable prompt_embeds so the CUE soft decoder can generate through this server (it sends
# precomputed inputs_embeds); harmless for base_local. Not valid for a pooling/embed server.
if [ "${ENABLE_PROMPT_EMBEDS:-1}" = "1" ] && [ -z "${CONVERT:-}" ]; then
  ARGS+=(--enable-prompt-embeds)
fi

echo "=== launching: CUDA_VISIBLE_DEVICES=${GPUS} vllm ${ARGS[*]} ==="
CUDA_VISIBLE_DEVICES="${GPUS}" "${VLLM_BIN}" "${ARGS[@]}" &
VLLM_PID=$!
trap 'kill ${VLLM_PID} 2>/dev/null || true' EXIT

BASE_URL="http://127.0.0.1:${PORT}/v1"
echo "=== waiting for ${BASE_URL}/models (up to ${WAIT_TIMEOUT}s) ==="
deadline=$(( $(date +%s) + WAIT_TIMEOUT ))
until curl -sf "${BASE_URL}/models" >/dev/null 2>&1; do
  if ! kill -0 "${VLLM_PID}" 2>/dev/null; then
    echo "ERROR: vLLM exited before becoming ready." >&2
    exit 1
  fi
  if [ "$(date +%s)" -ge "${deadline}" ]; then
    echo "ERROR: timed out waiting for vLLM after ${WAIT_TIMEOUT}s." >&2
    exit 1
  fi
  sleep 5
done

echo "=== vLLM ready at ${BASE_URL} (model: ${MODEL}) ==="
echo "Sidecar:  <cue-venv>/bin/python -m cue_training.evaluation.common.sidecar --vllm_base_url ${BASE_URL}"
wait "${VLLM_PID}"
