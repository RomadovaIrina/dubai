#!/bin/bash
# Agent C layer-3 GPU step for ONE video, inside the shared GPU lock (coordinator rule 2026-10-16): Chatterbox takes, then QA / CTC.
# VRAM capped at 6 GB per process in refit_offline.py; nvidia-smi sampled every second; wall time of the lock hold recorded.
set -u
id=$1; A=/workspace/dub/artifacts/audio-timing; R=$A/refit/$id
. $A/cpu_env.sh; unset CUDA_VISIBLE_DEVICES
echo "waiting for lock $(date +%T)"
flock /workspace/dub/gpu.lock bash -c "
  s=\$(date +%s.%N); echo \"lock acquired \$(date +%T)\"
  nvidia-smi --query-gpu=timestamp,memory.used,utilization.gpu --format=csv,noheader,nounits -l 1 > $R/nvsmi.log 2>/dev/null & NV=\$!
  nice -n 10 python scripts/quality/refit_offline.py tts --out $R; rc1=\$?
  nice -n 10 python scripts/quality/refit_offline.py qa --out $R; rc2=\$?
  kill \$NV 2>/dev/null; e=\$(date +%s.%N)
  python3 -c \"print('gpu_lock_wall_s', round(\$e - \$s, 1), 'rc', \$rc1, \$rc2)\" | tee $R/gpu_wall.txt
"
echo "lock released $(date +%T)"
