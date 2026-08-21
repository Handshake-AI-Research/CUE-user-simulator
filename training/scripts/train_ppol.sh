#!/bin/bash
set -euo pipefail
# Faithful PPol training: run the vendored Persona-Policies OpenEvolve pipeline to evolve the
# persona generator G(c, D, N) and freeze a best_program.py. Heavy (fitness runs live tau2
# episodes). Invoked by `uv run --group research cue-rollouts baseline ppol --train` via
# cue_training.baselines.common.manifest.train_paper; can also be run standalone.
#
# Default (standalone): evolve one generator per rollout simulator in parallel
# (llama / gpt / gemini). Each run gets its own PERSONA_POLICIES_VERSION,
# OpenEvolve yaml copy, and freeze under baselines/outputs/ppol/${DOMAIN}_${SIMTAG}/.
#
# Single-sim (manifest / orchestrator): PPOL_SINGLE=1 PPOL_SIM_MODEL=... ./scripts/train_ppol.sh
# Subset: PPOL_SIM_MODELS="gpt-5.4-mini,gemini/gemini-3.5-flash-lite" ./scripts/train_ppol.sh
#
# Pipeline (mirrors cue_training/baselines/ppol/external/persona-policies/persona_policies/README.md):
#   1. collect_baseline.py     -> baseline sim fingerprints (negatives)
#   2. train_discriminator.py  -> RF human-likeness discriminator (human ref = tau_bench_human.json,
#                                 which IS our tau-usi source: 150 airline + 345 retail)
#   3. run_evolution.py        -> curriculum N=5→8→10 + OpenEvolve mutations
#   4. (optional) benchmark    -> pick best-by-val-avg checkpoint (PPOL_BENCHMARK=1)
#
# Models (all via litellm): PPOL_SIM_MODEL is the tau2 user simulator used during fitness;
# PPOL_AGENT_MODEL the tau2 agent; PPOL_GEN_MODEL the persona generator/reflection LLM
# (paper: Gemini Flash — keep independent of PPOL_SIM_MODEL). Set PPOL_MUTATION_MODEL to
# also point OpenEvolve's code-mutation LLM at your provider (per-run temp yaml; never sed the
# shared vendored openevolve_config.yaml when running in parallel).
#
# Requires a dedicated venv with openevolve (the CUE 3.13 venv is incompatible); this script
# creates ${PPOL_VENV} (default .venv-ppol) and installs the vendored requirements + tau2-bench.

unset VIRTUAL_ENV UV_PROJECT_ENVIRONMENT
CUE_ROOT=$(cd "$(dirname "$0")/.." && pwd)
VENDOR="${CUE_ROOT}/cue_training/baselines/ppol/external/persona-policies"
TAU2_PATH="${CUE_ROOT}/cue_training/evaluation/tau2_bench/external/tau2-bench"

PPOL_DOMAIN=${PPOL_DOMAIN:-retail_airline}          # airline | retail | retail_airline
PPOL_AGENT_MODEL=${PPOL_AGENT_MODEL:-gpt-5.2}
PPOL_GEN_MODEL=${PPOL_GEN_MODEL:-openrouter/google/gemini-3-flash-preview}
# OpenEvolve's code-mutation LLM (openevolve_config.yaml). Defaults to PPOL_GEN_MODEL so it routes
# through our provider by default -- otherwise it stays the vendored OpenRouter default and 401s
# without an OpenRouter key. Set explicitly (e.g. an openrouter/ id) to override.
# Keep gen/mutator independent of PPOL_SIM_MODEL (paper decoupling; paper = Gemini Flash).
PPOL_MUTATION_MODEL=${PPOL_MUTATION_MODEL:-${PPOL_GEN_MODEL}}
PPOL_MUTATION_REASONING=${PPOL_MUTATION_REASONING:-low}
PPOL_ITERATIONS=${PPOL_ITERATIONS:-70}
PPOL_VENV=${PPOL_VENV:-${CUE_ROOT}/.venv-ppol}
PPOL_BASELINE_N=${PPOL_BASELINE_N:-100}

# Three rollout sims from configs/rollouts.json (api kind).
DEFAULT_PPOL_SIM_MODELS=(
  "openrouter/meta-llama/llama-3.1-8b-instruct"
  "gpt-5.4-mini"
  "gemini/gemini-3.5-flash-lite"
)

_simtag() {
  printf '%s' "$1" | tr '/:' '__'
}

# Match persona_policies.config._safe_path_segment (training_* folder names).
_safe_seg() {
  printf '%s' "$1" | sed 's/[^A-Za-z0-9_-]/_/g'
}

# Resolve which sims to train.
SIMS=()
if [ -n "${PPOL_SIM_MODELS:-}" ]; then
  # shellcheck disable=SC2206
  IFS=',' read -r -a _raw <<< "${PPOL_SIM_MODELS}"
  for _s in "${_raw[@]}"; do
    _s="$(printf '%s' "${_s}" | sed -e 's/^[[:space:]]*//' -e 's/[[:space:]]*$//')"
    [ -n "${_s}" ] && SIMS+=("${_s}")
  done
elif [ "${PPOL_SINGLE:-0}" = "1" ]; then
  SIMS+=("${PPOL_SIM_MODEL:-gpt-5.4-mini}")
else
  SIMS=("${DEFAULT_PPOL_SIM_MODELS[@]}")
fi
[ "${#SIMS[@]}" -gt 0 ] || { echo "ERROR: no PPOL_SIM_MODEL(S) to train." >&2; exit 1; }

[ -d "${VENDOR}" ] || { echo "ERROR: persona-policies missing at ${VENDOR}; clone it under cue_training/baselines/ppol/external/." >&2; exit 1; }
[ -d "${TAU2_PATH}/data/tau2/domains" ] || { echo "ERROR: tau2-bench task data missing at ${TAU2_PATH}/data/tau2/domains; clone tau2-bench under cue_training/evaluation/tau2_bench/external/." >&2; exit 1; }

# persona-policies resolves tau2 task data (split_tasks.json / tasks.json) via its default
# taubench_root = <persona-policies>/tau2-bench (not TAU2_DATA_DIR). Symlink it to our vendored
# tau2 checkout so the discriminator's official train/test split resolves.
# Re-link when the target is absent, broken, or points at a checkout without task data.
if [ ! -d "${VENDOR}/tau2-bench/data/tau2/domains" ]; then
  if [ -e "${VENDOR}/tau2-bench" ] && [ ! -L "${VENDOR}/tau2-bench" ]; then
    echo "ERROR: ${VENDOR}/tau2-bench exists but has no data/tau2/domains; remove it and re-run." >&2
    exit 1
  fi
  rm -f "${VENDOR}/tau2-bench"
  ln -s "${TAU2_PATH}" "${VENDOR}/tau2-bench"
  echo "=== linked ${VENDOR}/tau2-bench -> ${TAU2_PATH} ==="
fi

# 0. dedicated venv (openevolve + deps + tau2-bench). tau2-bench needs Python >=3.12,<3.14.
PPOL_PYVER=${PPOL_PYVER:-3.12}
if [ ! -x "${PPOL_VENV}/bin/python" ]; then
  echo "=== creating PPol venv ${PPOL_VENV} (Python ${PPOL_PYVER}) ==="
  uv venv --python "${PPOL_PYVER}" "${PPOL_VENV}"
fi
PPOL_PY="${PPOL_VENV}/bin/python"
if [ "${PPOL_INSTALL:-1}" = "1" ]; then
  echo "=== installing PPol deps into ${PPOL_VENV} ==="
  uv pip install --python "${PPOL_PY}" -r "${VENDOR}/persona_policies/requirements.txt"
  uv pip install --python "${PPOL_PY}" -e "${TAU2_PATH}"
fi

export TAU2_DATA_DIR=${TAU2_DATA_DIR:-${TAU2_PATH}/data}
# The vendored config also defaults the reflection + tau2 NL-assertion/env-interface models to
# OpenRouter. Route them through our provider too (honored by the sitecustomize patch even when
# the upstream checkout's config.py lacks the env hooks). PPOL_TAU2_UTIL_MODEL covers the two tau2 utils.
PPOL_TAU2_UTIL_MODEL=${PPOL_TAU2_UTIL_MODEL:-${PPOL_GEN_MODEL}}
export PERSONA_POLICIES_FEEDBACK_MODEL="${PPOL_GEN_MODEL}"
export PERSONA_POLICIES_NL_ASSERTIONS="${PPOL_TAU2_UTIL_MODEL}"
export PERSONA_POLICIES_ENV_INTERFACE="${PPOL_TAU2_UTIL_MODEL}"
# Vendored PersonaPoliciesConfig.curriculum=True with n_personas_schedule [(1,5),(2,8),(3,10)].
export PERSONA_POLICIES_CURRICULUM="${PERSONA_POLICIES_CURRICULUM:-1}"
export PERSONA_POLICIES_TAUBENCH_AGENT_MODEL="${PPOL_AGENT_MODEL}"
export PERSONA_POLICIES_LLM_MODEL="${PPOL_GEN_MODEL}"
export PERSONA_POLICIES_DOMAIN="${PPOL_DOMAIN}"

# Auto-register vLLM-served models with litellm at interpreter startup (sitecustomize on
# PYTHONPATH) so its cost/model-info lookups don't raise "This model isn't mapped yet".
PPOL_VLLM_MODELS=""
_collect_vllm() {
  case "$1" in hosted_vllm/*) PPOL_VLLM_MODELS="${PPOL_VLLM_MODELS:+${PPOL_VLLM_MODELS},}$1" ;; esac
}
_collect_vllm "${PPOL_GEN_MODEL}"
_collect_vllm "${PPOL_AGENT_MODEL}"
for _s in "${SIMS[@]}"; do _collect_vllm "${_s}"; done
export PPOL_VLLM_MODELS
export PYTHONPATH="${CUE_ROOT}/scripts/ppol_litellm_boot:${CUE_ROOT}:${VENDOR}:${PYTHONPATH:-}"

# Preflight: any hosted_vllm/ model needs HOSTED_VLLM_API_BASE and the exact served model id.
_preflight_vllm() {
  local _m="$1"
  case "${_m}" in
    hosted_vllm/*)
      [ -n "${HOSTED_VLLM_API_BASE:-}" ] || {
        echo "ERROR: ${_m} needs HOSTED_VLLM_API_BASE set (e.g. export HOSTED_VLLM_API_BASE=http://127.0.0.1:8000/v1)." >&2
        exit 1; }
      export HOSTED_VLLM_API_KEY=${HOSTED_VLLM_API_KEY:-EMPTY}
      local _served="${_m#hosted_vllm/}"
      if command -v curl >/dev/null 2>&1; then
        local _ids
        _ids=$(curl -sf "${HOSTED_VLLM_API_BASE%/}/models" 2>/dev/null || true)
        [ -n "${_ids}" ] || { echo "ERROR: no vLLM at HOSTED_VLLM_API_BASE=${HOSTED_VLLM_API_BASE} (/models unreachable)." >&2; exit 1; }
        case "${_ids}" in
          *"${_served}"*) : ;;
          *) echo "ERROR: vLLM at ${HOSTED_VLLM_API_BASE} does not serve '${_served}'. Served ids:" >&2
             printf '%s\n' "${_ids}" | grep -o '"id":"[^"]*"' >&2 || true
             echo "       Set PPOL_SIM_MODEL/PPOL_GEN_MODEL to hosted_vllm/<served-id>." >&2; exit 1 ;;
        esac
      fi ;;
  esac
}
_preflight_vllm "${PPOL_GEN_MODEL}"
_preflight_vllm "${PPOL_AGENT_MODEL}"
for _s in "${SIMS[@]}"; do _preflight_vllm "${_s}"; done

_train_one() {
  local sim_model="$1"
  local simtag version artifacts_dir best_program oe_cfg best logf
  simtag="$(_simtag "${sim_model}")"
  # Isolate OpenEvolve / training_* dirs per sim (never share training_${DOMAIN}/ across sims).
  # Sanitize like PersonaPoliciesConfig so BEST path matches on-disk training_<version>/.
  if [ "${PPOL_SINGLE:-0}" = "1" ] && [ -n "${PPOL_VERSION:-}" ]; then
    version="$(_safe_seg "${PPOL_VERSION}")"
  else
    version="$(_safe_seg "${PPOL_VERSION:-${PPOL_DOMAIN}_${simtag}}")"
  fi
  artifacts_dir="${ARTIFACTS_DIR:-${CUE_ROOT}/baselines/outputs/ppol/${PPOL_DOMAIN}_${simtag}}"
  # When ARTIFACTS_DIR was set for a single-sim invoke, keep it; for parallel fan-out ignore a
  # shared ARTIFACTS_DIR and always use the model-tagged path.
  if [ "${#SIMS[@]}" -gt 1 ]; then
    artifacts_dir="${CUE_ROOT}/baselines/outputs/ppol/${PPOL_DOMAIN}_${simtag}"
  fi
  best_program="${PPOL_BEST_PROGRAM:-${artifacts_dir}/best_program.py}"
  if [ "${#SIMS[@]}" -gt 1 ]; then
    best_program="${artifacts_dir}/best_program.py"
  fi
  mkdir -p "${artifacts_dir}"
  logf="${artifacts_dir}/train.log"

  # Per-run OpenEvolve config (parallel-safe; never mutate the shared vendored yaml).
  oe_cfg="${artifacts_dir}/openevolve_config.yaml"
  cp -f "${VENDOR}/persona_policies/evolution/openevolve_config.yaml" "${oe_cfg}"
  if [ -n "${PPOL_MUTATION_MODEL}" ]; then
    sed -i.tmp -E "s#^( *primary_model: ).*#\\1${PPOL_MUTATION_MODEL}#" "${oe_cfg}" && rm -f "${oe_cfg}.tmp"
    sed -i.tmp -E "s#^( *secondary_model: ).*#\\1${PPOL_MUTATION_MODEL}#" "${oe_cfg}" && rm -f "${oe_cfg}.tmp"
    sed -i.tmp -E "s#^( *reasoning_effort: ).*#\\1${PPOL_MUTATION_REASONING}#" "${oe_cfg}" && rm -f "${oe_cfg}.tmp"
  fi

  {
    echo "=== [${simtag}] collect baseline (domain=${PPOL_DOMAIN}, sim=${sim_model}) ==="
    PERSONA_POLICIES_VERSION="${version}" \
    PERSONA_POLICIES_TAUBENCH_USER_MODEL="${sim_model}" \
    PERSONA_POLICIES_OPENEVOLVE_CONFIG="${oe_cfg}" \
      "${PPOL_PY}" persona_policies/scripts/collect_baseline.py \
        --domain "${PPOL_DOMAIN}" --collect-split all --n "${PPOL_BASELINE_N}" ${PPOL_FORCE:+--force}

    echo "=== [${simtag}] train discriminator ==="
    PERSONA_POLICIES_VERSION="${version}" \
    PERSONA_POLICIES_TAUBENCH_USER_MODEL="${sim_model}" \
    PERSONA_POLICIES_OPENEVOLVE_CONFIG="${oe_cfg}" \
      "${PPOL_PY}" persona_policies/scripts/train_discriminator.py --domain "${PPOL_DOMAIN}"

    echo "=== [${simtag}] evolve (${PPOL_ITERATIONS} iters; curriculum N=5→8→10) ==="
    PERSONA_POLICIES_VERSION="${version}" \
    PERSONA_POLICIES_TAUBENCH_USER_MODEL="${sim_model}" \
    PERSONA_POLICIES_OPENEVOLVE_CONFIG="${oe_cfg}" \
      "${PPOL_PY}" persona_policies/evolution/run_evolution.py \
        --domain "${PPOL_DOMAIN}" --version "${version}" --iterations "${PPOL_ITERATIONS}" \
        ${PPOL_RESUME:+--resume}

    best="${VENDOR}/persona_policies/outputs/training_${version}/openevolve/best/best_program.py"
    if [ "${PPOL_BENCHMARK:-0}" = "1" ]; then
      echo "=== [${simtag}] benchmark (best-by-val-avg over N=5,8,10) ==="
      PERSONA_POLICIES_VERSION="${version}" \
      PERSONA_POLICIES_TAUBENCH_USER_MODEL="${sim_model}" \
        "${PPOL_PY}" -m persona_policies.benchmark --best-by-val-avg --n-personas-list 5,8,10 || \
          echo "WARN: [${simtag}] benchmark selection failed; using final best/best_program.py" >&2
    fi

    [ -f "${best}" ] || { echo "ERROR: [${simtag}] no best_program.py at ${best}." >&2; exit 1; }
    mkdir -p "${artifacts_dir}" "$(dirname "${best_program}")"
    cp -f "${best}" "${artifacts_dir}/best_program.py"
    cp -f "${best}" "${best_program}"

    # Shared "latest" pointer: mkdir-lock so parallel workers do not race (portable; no flock).
    mkdir -p "${CUE_ROOT}/baselines/outputs/ppol"
    _lockdir="${CUE_ROOT}/baselines/outputs/ppol/.latest.lockdir"
    while ! mkdir "${_lockdir}" 2>/dev/null; do sleep 0.05; done
    cp -f "${best}" "${CUE_ROOT}/baselines/outputs/ppol/best_program.py"
    printf '%s\n' "${best_program}" > "${CUE_ROOT}/baselines/outputs/ppol/LATEST"
    {
      echo "path=${best_program}"
      echo "provenance=${artifacts_dir}/best_program.py"
      echo "domain=${PPOL_DOMAIN}"
      echo "sim_model=${sim_model}"
      echo "gen_model=${PPOL_GEN_MODEL}"
      echo "mutation_model=${PPOL_MUTATION_MODEL}"
      echo "version=${version}"
      echo "source=${best}"
    } > "${CUE_ROOT}/baselines/outputs/ppol/LATEST.meta"
    rmdir "${_lockdir}"

    echo "=== [${simtag}] done. PPOL_BEST_PROGRAM=${best_program} ==="
  } 2>&1 | tee "${logf}"
}

echo "=== PPol train: ${#SIMS[@]} simulator(s) (parallel=$([[ ${#SIMS[@]} -gt 1 ]] && echo yes || echo no)) ==="
for _s in "${SIMS[@]}"; do echo "  - ${_s}"; done

cd "${VENDOR}"
PIDS=()
FAIL=0
for _s in "${SIMS[@]}"; do
  if [ "${#SIMS[@]}" -eq 1 ]; then
    _train_one "${_s}" || FAIL=1
  else
    _train_one "${_s}" &
    PIDS+=("$!")
  fi
done

if [ "${#PIDS[@]}" -gt 0 ]; then
  for _pid in "${PIDS[@]}"; do
    if ! wait "${_pid}"; then
      FAIL=1
    fi
  done
fi

if [ "${FAIL}" -ne 0 ]; then
  echo "ERROR: one or more PPol evolutions failed; check baselines/outputs/ppol/*/train.log" >&2
  exit 1
fi
echo "=== all PPol evolutions finished ==="
