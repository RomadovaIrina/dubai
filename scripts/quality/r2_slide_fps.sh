#!/bin/bash
# Round 2 track B: common slide metric at the SOURCE frame rate: output frame k vs source frame k (same instant), regions from MediaPipe
# landmarks at each source frame's TRUE time (timing/<id>_source_lm.npz). Window given in master frames (A:B) -> source frames inside it
# (2 master frames of margin: crossfades / window edges).
#   bash scripts/quality/r2_slide_fps.sh <id> <A:B> <out.json> <video.mp4>
set -u
id=$1; win=$2; out=$3; vid=$4
DUB_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"; export DUB_ROOT; . "${DUB_ROOT}/scripts/env.sh" >/dev/null; cd "${DUB_ROOT}"
R=/workspace/dub/artifacts/restoration-temporal/r2; SRC=$(ls /workspace/dub/inputs/test_videos_raw/$id.*)
K=$(python -c "
import numpy as np; z=np.load('$R/timing/${id}_source_lm.npz'); mu=z['mu']; A,B=[int(x) for x in '$win'.split(':')]
k=np.flatnonzero((mu>=A+2)&(mu<=B-3)); print(f'{k.min()}:{k.max()+1}')")
OPENCV_FOR_THREADS_NUM=8 python scripts/quality/slide_metric.py --video $vid --master $SRC --npz $R/timing/${id}_source_lm.npz --offset 0 --frames $K --out $out --series | cut -c1-300
