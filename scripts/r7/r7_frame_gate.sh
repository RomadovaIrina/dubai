#!/bin/bash
# r7 F: run the whole-frame gate (r7_frame_gate.py) on one version of all 5 videos.
#   bash scripts/r7/r7_frame_gate.sh <dir>:<label> [ids]     -> /workspace/dub/artifacts/r7f/frame_gate/<label>/<id>.json + <id>_sheet.jpg, summary line per id
# <dir>/<id>_<label>.mp4 must exist. Source landmarks: /workspace/dub/artifacts/r7f/lm/<id>_src_mp.npz (computed once: clean_face_stack.run_mediapipe on the source).
set -u
DUB_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"; export DUB_ROOT; . $DUB_ROOT/scripts/env.sh >/dev/null; cd $DUB_ROOT
D=${1%%:*}; L=${1##*:}; shift; IDS=${*:-03 01 05 02 04}; O=/workspace/dub/artifacts/r7f/frame_gate/$L; LM=/workspace/dub/artifacts/r7f/lm; mkdir -p $O $LM
for id in $IDS; do
  src=$(ls /workspace/dub/inputs/test_videos_raw/$id.* | head -1); v=$D/${id}_$L.mp4; [ -f $v ] || { echo "== frame_gate $L $id MISSING $v"; continue; }
  [ -f $LM/${id}_src_mp.npz ] || python -c "import sys,pathlib; sys.path.insert(0,'scripts/pilot'); import clean_face_stack as c; c.run_mediapipe(pathlib.Path('$src'), pathlib.Path('$LM/${id}_src_mp.npz'))"
  python scripts/r7/r7_frame_gate.py --src $src --out-video $v --lm $LM/${id}_src_mp.npz --out $O/$id.json --label $L --id $id --sheet $O/${id}_sheet.jpg ${GATE_ARGS:-}
done
