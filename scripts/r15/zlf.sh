#!/bin/bash
# r15 Z: agent F's lip_flicker.py (excess lip "blinks" / 10 s at a steady shape, the source's own as the floor) on a W arm final
# (scripts/r15/arm.sh: eval/<arm>_<id>/<arm>_<id>.mp4, generator run r9g_q2_<id>_<gen arm>) -> eval/lipflick.jsonl
#   bash scripts/r15/zlf.sh <id> <arm> [<generator arm, default = arm>]
set -u; id=$1; arm=$2; gen=${3:-$2}
cd "$(dirname "${BASH_SOURCE[0]}")/../.."; . scripts/env.sh >/dev/null
X=/workspace/dub/r15w/eval; B=/workspace/dub/artifacts/r4m; ID=q2_$id; GT=r9g_${ID}_$gen; W=$B/$GT/work_${ID}_$GT
V=$X/${arm}_$id/${arm}_$id.mp4; SRC=$(ls $X/$ID/e2e/work/in/$ID.* | head -1)
python scripts/r15/lip_flicker.py --video $V --lm $W/${ID}_${GT}_25fps_ls_raw.mediapipe.npz --timing $B/$GT/timing/${ID}_master_timing_F.json \
  --master $W/video/master_25fps.mp4 --src $SRC --src-lm $B/$GT/timing/${ID}_source_lm_F.npz --label ${arm}_$id --json $X/lipflick.jsonl 2>&1 | grep -vE "^W0000|^I0000|^INFO|^WARNING" | tail -2
