#!/bin/bash
# Round 4 V queue 2 (task 3): LatentSync directly at the source rate (30 fps master = the source frames, no 25->30 retime), on v4-repro's audio.
set -u
DUB_ROOT=/workspace/dub/dubai; . $DUB_ROOT/scripts/env.sh >/dev/null; cd $DUB_ROOT
B=/workspace/dub/artifacts/r4v; A=$B/audio-r4b
until grep -q V4REPRO_DONE $B/v4repro.out; do sleep 30; done
for id in 04 03 05 01 02; do
  LS_FPS=30 bash scripts/r4/r4_video.sh $id $A/$id $A/$id/${id}_r4b.manifest.json $B/ls30 ls30
done
cp -n $B/ls30/timing/* $B/v4repro/timing/ 2>/dev/null
for id in 04 03 05 01 02; do python scripts/r4/r4_gate.py score --label ls30 --dir $B/ls30 --ids $id; done
echo Q2_DONE
