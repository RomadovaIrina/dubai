#!/bin/bash
# DabAI worker entrypoint (r15 T). Modes (the container's command):
#   api                  (default) the HTTP job API on :${PORT:-8000} (scripts/job/job_api.py), one job at a time on this GPU
#   job <job.json>       one job through the CLI (scripts/job/dabai_job.py), exit 0 = SUCCESS / PARTIAL_SUCCESS
#   fetch-models         fill the models volume once (+ the NC volume when DUB_PROFILE=current); needs network + HF_TOKEN
#   check [--deep]       weights / GPU / venv self-test (infra/docker/check.sh); --deep also hashes the weight files
#   smoke <video.mp4>    one real job on a short clip (~5-10 min on a 5090) -> /data/jobs/smoke/status.json
#   anything else        exec'd as is (e.g. bash)
# DUB_PROFILE (required; baked at build, -e overrides): clean = DUB_ALLOW_NC=0 DUB_ALLOW_NC_BED=0 | current = 1 / 1 + the NC volume.
set -euo pipefail
R=${DUB_ROOT:-/workspace/dub/dubai}; M=$R/models; N=${DABAI_NC_DIR:-/opt/dabai-nc}; mode=${1:-api}; [ $# -gt 0 ] && shift
echo "$mode" > /tmp/dabai_mode

case "${DUB_PROFILE:-}" in   # no silent default: the profile is a product / licence decision (INTERFACE.md section 8)
  clean)   export DUB_ALLOW_NC=0 DUB_ALLOW_NC_BED=0 ;;
  current) export DUB_ALLOW_NC=1 DUB_ALLOW_NC_BED=1 ;;
  *)       echo "DUB_PROFILE must be 'clean' or 'current' (got '${DUB_PROFILE:-}'): an explicit product / licence decision, see INTERFACE.md section 8" >&2; exit 64 ;;
esac

link_runtime() {   # caches + checkpoints from the models volume (no download at start, TZ 5.1); the NC parts from their own volume (current only)
  mkdir -p /root/.cache/torch
  [ -d $M/cache/pyannote ] && ln -sfn $M/cache/pyannote /root/.cache/torch/pyannote
  [ -d $M/cache/pkuseg ] && ln -sfn $M/cache/pkuseg /root/.pkuseg
  local LS=$R/third_party/latentsync/checkpoints; mkdir -p $LS/auxiliary
  ln -sfn $M/latentsync/latentsync_unet.pt $LS/latentsync_unet.pt; ln -sfn $M/latentsync/whisper $LS/whisper
  for f in syncnet_v2.model sfd_face.pth; do ln -sfn $M/latentsync/auxiliary/$f $LS/auxiliary/$f; done
  if [ -d ${KS_ROOT:-/opt/keysync}/code ]; then   # r16: KeySync weights from the volume (fetch-models 8/8)
    mkdir -p ${KS_ROOT:-/opt/keysync}/code/pretrained_models; ln -sfn $M/keysync/checkpoints ${KS_ROOT:-/opt/keysync}/code/pretrained_models/checkpoints
  fi
  if [ "$DUB_PROFILE" = current ]; then
    for p in CodeFormer/weights/CodeFormer/codeformer.pth quarantine_nc/torch/hub/checkpoints/955717e8-8726e21a.th sep_pylib/demucs; do
      [ -e $N/$p ] || { echo "DUB_PROFILE=current needs the NC volume at $N ($p missing): run fetch-models with DUB_PROFILE=current and -v <nc volume>:$N" >&2; exit 64; }
    done
    echo "WARNING: DUB_PROFILE=current: non-commercial components (CodeFormer S-Lab NC, htdemucs weights) are ENABLED from $N: a deliberate licence deviation" >&2
    ln -sfn $N/CodeFormer $R/third_party/CodeFormer; ln -sfn $N/quarantine_nc /workspace/dub/quarantine_nc; ln -sfn $N/sep_pylib /workspace/dub/sep_pylib
  fi
  return 0
}

case "$mode" in
  api)          link_runtime
                if [ "${DABAI_RESIDENT:-0}" = 1 ]; then   # r17: the resident worker (models loaded once, parked between stages) starts beside the API;
                  mkdir -p ${DABAI_RESIDENT_DIR:-/data/resident}   # jobs posted before it is ready run cold (/health: resident.state)
                  ( cd $R && bash scripts/job/resident.sh start > ${DABAI_RESIDENT_DIR:-/data/resident}/start.log 2>&1 ) &
                fi
                cd $R/scripts/job; exec /venv/api/bin/python -m uvicorn job_api:app --host 0.0.0.0 --port "${PORT:-8000}" --workers 1 ;;
  job)          link_runtime; exec /venv/api/bin/python $R/scripts/job/dabai_job.py "$@" ;;
  fetch-models) exec bash $R/infra/docker/fetch_models.sh "$@" ;;
  check)        link_runtime; exec bash $R/infra/docker/check.sh "$@" ;;
  smoke)        link_runtime; exec /venv/api/bin/python $R/scripts/job/dabai_job.py --input "${1:?smoke <video.mp4>}" --out /data/jobs/smoke --job-id smoke ;;
  *)            link_runtime; exec "$mode" "$@" ;;
esac
