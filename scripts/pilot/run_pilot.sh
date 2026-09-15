#!/usr/bin/env bash
# Thin dispatcher for the pilot toolkit. It does not hide required inputs.
set -euo pipefail
ROOT="${DUB_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
PY="${DUB_ENV:-/venv/dabai}/bin/python"
cd "$ROOT"
T="scripts/pilot"
TASK="${1:-status}"; shift || true
case "$TASK" in
  0.1) "$PY" "$T/validate_01_blackwell.py" --run-smoke "$@" ;;
  0.2) "$PY" "$T/validate_02_latentsync.py" --run-smoke "$@" ;;
  0.3) bash "$T/probe_03_flashattention_build.sh" "$@" ;;
  0.4-peaks) "$PY" "$T/run_vram_suite.py" "$@" ;;
  0.4) "$PY" "$T/resident_vram_probe.py" "$@" ;;
  0.5) "$PY" "$T/benchmark_05_gpu_coefficient.py" "$@" ;;
  0.6) "$PY" "$T/benchmark_06_codeformer.py" "$@" ;;
  0.7) "$PY" "$T/benchmark_07_codeformer_syncnet.py" "$@" ;;
  0.8) "$PY" "$T/benchmark_08_syncnet_dataset.py" "$@" ;;
  0.9-fetch) "$PY" "$T/fetch_09_qwen_quant.py" "$@" ;;
  0.9) "$PY" "$T/benchmark_09_qwen_quant.py" "$@" ;;
  0.10) "$PY" "$T/benchmark_10_cold_start.py" "$@" ;;
  0.11) "$PY" "$T/benchmark_11_s_dataset.py" "$@" ;;
  status) "$PY" "$T/pilot_status.py" ;;
  *) echo "usage: $0 {0.1|0.2|0.3|0.4-peaks|0.4|0.5|0.6|0.7|0.8|0.9-fetch|0.9|0.10|0.11|status} [args...]"; exit 2;;
esac
