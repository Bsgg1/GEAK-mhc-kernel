#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

export GEAK_OFFLOAD_ARCH="${GEAK_OFFLOAD_ARCH:-gfx928}"
export MHC_PRE_FORCE_REBUILD="${MHC_PRE_FORCE_REBUILD:-1}"
export PYTHONPATH="${ROOT}/baseline:${PYTHONPATH:-}"

exec python3 "${ROOT}/baseline/test_mhc_pre_hip_harness.py" "$@"
