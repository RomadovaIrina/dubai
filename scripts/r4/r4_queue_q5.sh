#!/bin/bash
# Round 4 V queue 5: r4_video_v1 (v4 stack minus lip sharpen and seam-LF) re-composited from the v4-repro raw layers (video-only, v4-repro audio)
# and from the audio-v1 raw layers (merged candidate m1), then gate scores.
set -u
DUB_ROOT=/workspace/dub/dubai; . $DUB_ROOT/scripts/env.sh >/dev/null; cd $DUB_ROOT
B=/workspace/dub/artifacts/r4v
for id in 04 03 05 01 02; do OUT_DIR=$B/v1 FIX30_THREADS=8 bash scripts/r4/r4_ablate.sh $id v1 & done; wait
for id in 04 03 05 01 02; do OUT_DIR=$B/m1c RUN_DIR=$B/m1 RUN_TAG=m1v4 FIX30_THREADS=8 bash scripts/r4/r4_ablate.sh $id v1 & done
for id in 04 03 05 01 02; do echo "v1 $B/v1 $id --asr-from $B/v4repro:v4repro"; done | xargs -P 3 -L 1 bash -c 'python scripts/r4/r4_gate.py score --label $0 --dir $1 --ids $2 $3 $4'
wait
for id in 04 03 05 01 02; do echo "v1 $B/m1c $id"; echo "m1v4 $B/m1 $id"; done | xargs -P 3 -L 1 bash -c 'python scripts/r4/r4_gate.py score --label $0 --dir $1 --ids $2'
echo Q5_DONE
