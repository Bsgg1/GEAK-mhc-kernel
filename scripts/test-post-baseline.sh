#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

export GEAK_OFFLOAD_ARCH="${GEAK_OFFLOAD_ARCH:-gfx928}"
export MHC_POST_FORCE_REBUILD="${MHC_POST_FORCE_REBUILD:-1}"
export PYTHONPATH="${ROOT}/examples/mhc_ops:${PYTHONPATH:-}"

exec python3 "${ROOT}/examples/mhc_ops/test_mhc_post_hip_harness.py" "$@"
