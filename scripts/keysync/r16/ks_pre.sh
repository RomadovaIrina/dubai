#!/bin/bash
# r17 speed, output-identical: the KeySync master (the source at 25 fps) and its MediaPipe landmarks do not depend on the dub, so e2e.sh starts
# them in the background at the audio stage (CPU work hidden behind the GPU audio). The SAME commands as ks_face.sh's prep (master: libx264
# crf 10 fast; landmarks: scripts/pilot/mp_landmarks.py, VIDEO mode, OMP 4) -> ks_face.sh copies them instead of computing them (KS_PRE_DIR).
#     bash ks_pre.sh <src video> <dir>   -> <dir>/master_25fps.mp4, <dir>/master_mp.npz, <dir>/done (or <dir>/failed)
R=$(cd "$(dirname "$0")/../../.." && pwd); SRC=$1; D=$2; MPY=${DUB_MEDIAPIPE_PY:-/venv/mediapipe/bin/python}; mkdir -p $D; rm -f $D/done $D/failed
t0=$(date +%s.%N)
ffmpeg -v error -y -i "$SRC" -map 0:v:0 -an -vf fps=25 -c:v libx264 -preset fast -crf 10 -pix_fmt yuv420p $D/master_25fps.mp4 \
  && GLOG_minloglevel=2 OMP_NUM_THREADS=4 $MPY $R/scripts/pilot/mp_landmarks.py $D/master_25fps.mp4 $D/master_mp.npz --num-faces ${KS_NUM_FACES:-4} > $D/mp.log 2>&1 \
  && { echo "$(python3 -c "import time; print(round(time.time() - $t0, 1))")" > $D/done; exit 0; }
touch $D/failed; exit 1
