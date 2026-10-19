#!/bin/bash
# Round 2 (track A): expose the canonical baseline work dir of <id> at the path its manifest records
# (/workspace/dub/artifacts/baseline/<id>/work/work_<id>) as symlinks, plus the regenerated 25 fps master, its MediaPipe pass and rigid motion.
set -e
id=$1; C=/workspace/dub/canonical/artifacts/baseline/$id/work/work_$id; D=/workspace/dub/artifacts/baseline/$id/work/work_$id
M=/workspace/dub/artifacts/lipsync-temporal/r2/masters/$id; R=/workspace/dub/artifacts/lipsync-temporal/r2/anchor/${id}_rigid.npz
mkdir -p $D
for f in $C/*; do ln -sfn $f $D/; done
ln -sfn $M/master_25fps.mp4 $D/master_25fps.mp4; ln -sfn $M/master_mediapipe.npz $D/master_mediapipe.npz
[ -f $R ] && ln -sfn $R $D/master_rigid.npz
ls $D | wc -l
