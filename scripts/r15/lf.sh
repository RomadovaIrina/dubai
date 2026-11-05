#!/bin/bash
# r15 F3: lip_flicker.py on a final by its composite tag (r9f_<tag>): resolves the generator run, its layer landmarks, timing and source.
#   bash scripts/r15/lf.sh <tag> [json]
set -u; tag=$1; J=${2:-/workspace/dub/r15f3/lipflick.jsonl}; cd "$(dirname "${BASH_SOURCE[0]}")/../.."; . scripts/env.sh >/dev/null
B=/workspace/dub/artifacts/r4m; V=$(ls $B/r9f_$tag/*_r9f_$tag.mp4 | head -1); id=$(basename $V); id=${id%%_r9f_*}
GT=$(grep -o "from [A-Za-z0-9_]* layers" $B/r9f_$tag/README.txt 2>/dev/null | tail -1 | awk '{print $2}'); GT=${GT:-${GT_OVR:-r9g_$id}}
W=$B/$GT/work_${id}_$GT; v=${id#*_}; SRC=$(ls /workspace/dub/r14d/eval/q2_$v/e2e/work/in/q2_$v.* | head -1)
python scripts/r15/lip_flicker.py --video $V --lm $W/${id}_${GT}_25fps_ls_raw.mediapipe.npz --timing $B/$GT/timing/${id}_master_timing_F.json \
  --master $W/video/master_25fps.mp4 --src $SRC --src-lm $B/$GT/timing/${id}_source_lm_F.npz --label $tag --json $J
