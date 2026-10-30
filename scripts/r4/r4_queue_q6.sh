#!/bin/bash
# Round 4 V queue 6: task 4 (reproduce v5's "half face goes dark" on 04 with v5's own one-command runner, measure it), task 5 chin-clip arms
# (CPU composites of v1 + --chin-clip 0.3 / 0.5 on the v4-repro raw layers) + scores.
set -u
DUB_ROOT=/workspace/dub/dubai; . $DUB_ROOT/scripts/env.sh >/dev/null; cd $DUB_ROOT
B=/workspace/dub/artifacts/r4v
for id in 04 03 05 01; do (OUT_DIR=$B/chin FIX30_THREADS=8 bash scripts/r4/r4_ablate.sh $id chin3 chin5) & done
until grep -q Q2_DONE $B/q2.out; do sleep 30; done
mkdir -p $B/v5; flock /workspace/dub/gpu.lock bash scripts/r3_e2e_v5.sh 04 $B/v5 > $B/v5/run.out 2>&1
python scripts/r4/r4_face_metrics.py --video $B/v5/04_e2e_v5.mp4 --src ${DUB_INPUTS:-/workspace/dub/inputs/test_videos_raw}/04.mp4 --src-lm $B/v4repro/timing/04_source_lm_F.npz --out $B/v5/04_e2e_v5.r4face.json > $B/v5/face.out 2>&1
wait
for id in 04 03 05 01; do for arm in chin3 chin5; do echo "$arm $B/chin $id"; done; done | xargs -P 3 -L 1 bash -c 'python scripts/r4/r4_gate.py score --label $0 --dir $1 --ids $2 --asr-from '$B'/v4repro:v4repro'
echo Q6_DONE
