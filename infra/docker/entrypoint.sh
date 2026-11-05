#!/bin/bash
# DabAI worker entrypoint (r15 T). Modes (the container's command):
#   api                  (default) the HTTP job API on :${PORT:-8000} (scripts/job/job_api.py), one job at a time on this GPU
#   job <job.json>       one job through the CLI (scripts/job/dabai_job.py), exit 0 = SUCCESS / PARTIAL_SUCCESS
#   fetch-models         fill the models volume once (infra/docker/fetch_models.sh; needs HF_TOKEN for the gated pyannote 3.1, network)
#   check [--deep]       weights / GPU / venv self-test (infra/docker/check.sh); --deep also hashes the weight files
#   smoke <video.mp4>    one real job on a short clip (~5-10 min on a 5090) -> /data/jobs/smoke/status.json
#   anything else        exec'd as is (e.g. bash)
set -euo pipefail
R=${DUB_ROOT:-/workspace/dub/dubai}; M=$R/models; mode=${1:-api}; [ $# -gt 0 ] && shift
echo "$mode" > /tmp/dabai_mode

link_runtime() {   # caches + checkpoints that must come from the models volume (no download at start, TZ 5.1)
  mkdir -p /root/.cache/torch
  [ -d $M/cache/pyannote ] && ln -sfn $M/cache/pyannote /root/.cache/torch/pyannote
  [ -d $M/cache/pkuseg ] && ln -sfn $M/cache/pkuseg /root/.pkuseg
  local LS=$R/third_party/latentsync/checkpoints; mkdir -p $LS/auxiliary
  ln -sfn $M/latentsync/latentsync_unet.pt $LS/latentsync_unet.pt; ln -sfn $M/latentsync/whisper $LS/whisper
  for f in syncnet_v2.model sfd_face.pth; do ln -sfn $M/latentsync/auxiliary/$f $LS/auxiliary/$f; done
  if [ "${DUB_ALLOW_NC:-0}" = 1 ] || [ "${DUB_ALLOW_NC_BED:-0}" = 1 ]; then   # the NC profile: only from an operator-mounted volume
    local N=${DABAI_NC_DIR:-/opt/dabai-nc}
    [ -d $N ] || { echo "DUB_ALLOW_NC / DUB_ALLOW_NC_BED = 1 but $N (CodeFormer, quarantine_nc, sep_pylib) is not mounted" >&2; exit 64; }
    echo "WARNING: non-commercial components enabled (DUB_ALLOW_NC=${DUB_ALLOW_NC:-0} DUB_ALLOW_NC_BED=${DUB_ALLOW_NC_BED:-0}) from $N: a deliberate licence deviation" >&2
    [ "${DUB_ALLOW_NC:-0}" = 1 ] && ln -sfn $N/CodeFormer $R/third_party/CodeFormer
    [ "${DUB_ALLOW_NC_BED:-0}" = 1 ] && { ln -sfn $N/quarantine_nc /workspace/dub/quarantine_nc; ln -sfn $N/sep_pylib /workspace/dub/sep_pylib; }
  fi
  return 0
}

case "$mode" in
  api)          link_runtime; cd $R/scripts/job; exec /venv/api/bin/python -m uvicorn job_api:app --host 0.0.0.0 --port "${PORT:-8000}" --workers 1 ;;
  job)          link_runtime; exec /venv/api/bin/python $R/scripts/job/dabai_job.py "$@" ;;
  fetch-models) exec bash $R/infra/docker/fetch_models.sh "$@" ;;
  check)        link_runtime; exec bash $R/infra/docker/check.sh "$@" ;;
  smoke)        link_runtime; exec /venv/api/bin/python $R/scripts/job/dabai_job.py --input "${1:?smoke <video.mp4>}" --out /data/jobs/smoke --job-id smoke ;;
  *)            link_runtime; exec "$mode" "$@" ;;
esac
