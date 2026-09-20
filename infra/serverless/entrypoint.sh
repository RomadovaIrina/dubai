#!/usr/bin/env bash
# DabAI RunPod Serverless worker entrypoint.
#
# Responsibilities (Pilot 0.10 topology: multi-stage image + weights on the network volume):
#   1. fail fast if the network volume / model root is absent (never download anything);
#   2. map the volume into the layout the DabAI code expects (idempotent symlinks);
#   3. export the offline environment (belt and braces: the image already sets it);
#   4. start the RunPod handler.
#
# Usage:  entrypoint.sh [serve [handler args...] | prepare | check | info | <command ...>]
#   serve   (default) start the RunPod Serverless worker (python handler.py ...)
#   prepare only steps 1-3, exit 0 — used by the local smoke test
#   check   steps 1-3 then print the readiness JSON and exit 0/1
#   info    steps 1-3 then print image/version info
set -euo pipefail

VOL="${DABAI_VOLUME:-/runpod-volume}"
export MODELS_ROOT="${MODELS_ROOT:-${VOL}/models}"
export DUB_ROOT="${DUB_ROOT:-/opt/dabai}"
HANDLER="${DUB_ROOT}/infra/serverless/handler.py"
PYTHON="${DUB_ENV:-/venv/dabai}/bin/python"

log() { printf '[entrypoint %s] %s\n' "$(date -u +%H:%M:%S)" "$*" >&2; }
die() { printf '[entrypoint] FATAL: %s\n' "$*" >&2; exit 64; }

# --- 1. network volume ----------------------------------------------------------------------------------------------
[ -d "${VOL}" ] || die "network volume is not mounted at ${VOL} (attach the dabai-models volume to the endpoint)"
[ -d "${MODELS_ROOT}" ] || die "${MODELS_ROOT} does not exist on the volume (expected the layout from MODEL_VOLUME_MANIFEST.json)"
if [ -f "${MODELS_ROOT}/MODEL_VOLUME_MANIFEST.json" ]; then
  log "model root ${MODELS_ROOT} (manifest present)"
else
  log "WARNING: ${MODELS_ROOT}/MODEL_VOLUME_MANIFEST.json is missing; continuing, the readiness check lists what is absent"
fi

# --- 2. environment (offline) ---------------------------------------------------------------------------------------
export HF_HOME="${MODELS_ROOT}/hf"
export HF_HUB_OFFLINE=1
export HF_HUB_DISABLE_TELEMETRY=1
export PYANNOTE_CACHE="${MODELS_ROOT}/hf/hub"
export PKUSEG_HOME="${MODELS_ROOT}/pkuseg"
export TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD="${TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD:-1}"
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"
export MPLBACKEND="${MPLBACKEND:-Agg}"

# --- 3. idempotent symlinks -----------------------------------------------------------------------------------------
# link <target> <link>: creates/updates <link> -> <target>; refuses to destroy real data.
link() {
  local target="$1" lnk="$2"
  [ -e "${target}" ] || die "symlink target missing on the volume: ${target}"
  if [ -L "${lnk}" ]; then
    if [ "$(readlink -f "${lnk}")" != "$(readlink -f "${target}")" ]; then
      ln -sfn "${target}" "${lnk}"; log "relinked ${lnk} -> ${target}"
    else
      log "ok ${lnk} -> ${target}"
    fi
  elif [ -d "${lnk}" ]; then
    rmdir "${lnk}" 2>/dev/null || die "${lnk} is a non-empty directory; refusing to replace it with a symlink"
    ln -s "${target}" "${lnk}"; log "linked ${lnk} -> ${target}"
  elif [ -e "${lnk}" ]; then
    die "${lnk} exists and is neither a symlink nor a directory"
  else
    mkdir -p "$(dirname "${lnk}")"
    ln -s "${target}" "${lnk}"; log "linked ${lnk} -> ${target}"
  fi
}

# DabAI code: pilot_common.MODELS = <DUB_ROOT>/models
link "${MODELS_ROOT}" "${DUB_ROOT}/models"
# LatentSync (cwd = clone root at inference): checkpoints/{latentsync_unet.pt, whisper/tiny.pt} and the insightface
# root "checkpoints/auxiliary" -> models/buffalo_l/*.onnx (latentsync/utils/face_detector.py; ~/.insightface unused)
LS_CKPT="${DUB_ROOT}/third_party/latentsync/checkpoints"
mkdir -p "${LS_CKPT}"
link "${MODELS_ROOT}/latentsync/latentsync_unet.pt" "${LS_CKPT}/latentsync_unet.pt"
link "${MODELS_ROOT}/latentsync/whisper"            "${LS_CKPT}/whisper"
link "${MODELS_ROOT}/insightface"                   "${LS_CKPT}/auxiliary"

# --- 4. run ---------------------------------------------------------------------------------------------------------
mode="${1:-serve}"
case "${mode}" in
  prepare) log "prepare done"; exit 0 ;;
  check)   exec "${PYTHON}" "${HANDLER}" --check ;;
  info)    exec "${PYTHON}" "${HANDLER}" --info ;;
  serve)   shift || true; log "starting RunPod handler"; exec "${PYTHON}" -u "${HANDLER}" "$@" ;;
  *)       exec "$@" ;;
esac
