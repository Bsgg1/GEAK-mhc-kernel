#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
GEAK_ROOT="${GEAK_ROOT:-/root/GEAK}"
GEAK_CONFIG="${GEAK_CONFIG:-${GEAK_ROOT}/config/local/hygon_k500sm_gfx928_codex_openai.yaml}"
OUT_DIR="${OUT_DIR:-${GEAK_ROOT}/optimization_logs/mhc_pre_cu_hygon_opt_from_repo}"

export GEAK_USE_CODEX_OPENAI_KEY="${GEAK_USE_CODEX_OPENAI_KEY:-1}"
export GEAK_BASELINE_REPEATS="${GEAK_BASELINE_REPEATS:-1}"
export GEAK_PROFILE_TIMEOUT="${GEAK_PROFILE_TIMEOUT:-15}"
export GEAK_BENCH_TIMEOUT="${GEAK_BENCH_TIMEOUT:-180}"
export GEAK_LLM_REQUEST_TIMEOUT="${GEAK_LLM_REQUEST_TIMEOUT:-180}"
export MSWEA_MODEL_RETRY_STOP_AFTER_ATTEMPT="${MSWEA_MODEL_RETRY_STOP_AFTER_ATTEMPT:-3}"
export GEAK_OFFLOAD_ARCH="${GEAK_OFFLOAD_ARCH:-gfx928}"
export MHC_PRE_FORCE_REBUILD="${MHC_PRE_FORCE_REBUILD:-1}"

if [[ -f "${GEAK_ROOT}/scripts/geak-hygon-env.sh" ]]; then
  # shellcheck source=/dev/null
  source "${GEAK_ROOT}/scripts/geak-hygon-env.sh"
fi

geak --config "${GEAK_CONFIG}" \
  --repo "${REPO_ROOT}/examples/mhc_ops" \
  --kernel-url "${REPO_ROOT}/examples/mhc_ops/src/mhc_pre.cu" \
  --test-command "python3 ${REPO_ROOT}/examples/mhc_ops/test_mhc_pre_hip_harness.py --full-benchmark" \
  --task "Optimize the CUDA/HIP mhc_pre baseline kernel at ${REPO_ROOT}/examples/mhc_ops/src/mhc_pre.cu for Hygon K500SM_AI DCU gfx928. The source is compiled by hipcc with -x hip. Metric is GEAK_RESULT_LATENCY_MS, lower is better. Preserve correctness for all harness shapes. Precision requirements: residual is BF16, fn/hc_scale/hc_base are FP32, GEMM accumulation is FP32, post_mix and comb_mix are FP32, and layer_input output is BF16." \
  --gpu-ids 0 \
  --num-parallel 1 \
  --debug \
  --yolo \
  --exit-immediately \
  -o "${OUT_DIR}"
