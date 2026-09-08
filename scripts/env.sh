#!/bin/bash
# source this file: activates the dabai env and sets project-wide variables.
export DUB_ROOT="${DUB_ROOT:-/workspace/dub}"
export DUB_ENV="${DUB_ENV:-/venv/dabai}"
# shellcheck disable=SC1091
[ -f "${DUB_ROOT}/../.env" ] && set -a && . "${DUB_ROOT}/../.env" && set +a
export HF_HOME="${HF_HOME:-${DUB_ROOT}/models/hf}"
export HF_HUB_DISABLE_TELEMETRY=1
export TOKENIZERS_PARALLELISM=false
# cgroup quota is ~16 vCPU; os.cpu_count() reports the host's 128 cores, so pin threads explicitly
export DUB_THREADS="${DUB_THREADS:-16}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-${DUB_THREADS}}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-${DUB_THREADS}}"
# CodeFormer is run from its clone (vendored basicsr + facelib), LatentSync via `python -m scripts.inference` from its root
export PYTHONPATH="${DUB_ROOT}/third_party/CodeFormer${PYTHONPATH:+:${PYTHONPATH}}"
export PATH="${DUB_ENV}/bin:${PATH}"
export VIRTUAL_ENV="${DUB_ENV}"
