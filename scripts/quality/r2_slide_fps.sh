#!/bin/bash
# Round 2 track B: common slide metric at the SOURCE frame rate: output frame k vs source frame k (same instant), regions from MediaPipe
# landmarks at each source frame's TRUE time (timing/<id>_source_lm.npz). Window given in master frames (A:B) -> source frames inside it
# (2 master frames of margin). Both videos are cut frame-exactly first (r2_cut_frames.py: OpenCV seeking is not frame-exact here).
#   bash scripts/quality/r2_slide_fps.sh <id> <A:B> <out.json> <video.mp4> [position-check out.json]
set -u
id=$1; win=$2; out=$3; vid=$4; pos=${5:-}
DUB_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"; export DUB_ROOT; . "${DUB_ROOT}/scripts/env.sh" >/dev/null; cd "${DUB_ROOT}"
R=/workspace/dub/artifacts/restoration-temporal/r2; SRC=$(ls /workspace/dub/inputs/test_videos_raw/$id.*); C=$R/fps30/cuts; mkdir -p $C
K=$(python -c "
import numpy as np; z=np.load('$R/timing/${id}_source_lm.npz'); mu=z['mu']; A,B=[int(x) for x in '$win'.split(':')]
k=np.flatnonzero((mu>=A+2)&(mu<=B-3)); print(f'{k.min()}:{k.max()+1}')")
[ -f $C/${id}_src_${K/:/_}.mp4 ] || python scripts/quality/r2_cut_frames.py $SRC $K $C/${id}_src_${K/:/_}.mp4 --npz $R/timing/${id}_source_lm.npz --npz-out $C/${id}_lm_${K/:/_}.npz > /dev/null
tmpv=$C/.cut_$(basename $vid .mp4)_${K/:/_}.mp4; python scripts/quality/r2_cut_frames.py $vid $K $tmpv > /dev/null
N=$(( ${K#*:} - ${K%:*} ))
OPENCV_FOR_THREADS_NUM=8 python scripts/quality/slide_metric.py --video $tmpv --master $C/${id}_src_${K/:/_}.mp4 --npz $C/${id}_lm_${K/:/_}.npz --offset 0 --frames 0:$N --out $out --series | cut -c1-300
[ -n "$pos" ] && OPENCV_FOR_THREADS_NUM=4 python scripts/quality/r2_position_check.py --video $tmpv --master $C/${id}_src_${K/:/_}.mp4 --npz $C/${id}_lm_${K/:/_}.npz --offset 0 --frames 0:$N --out $pos | cut -c1-300
python -c "import json;d=json.load(open('$out'));d['source_frames']='$K';json.dump(d,open('$out','w'),indent=1)"
