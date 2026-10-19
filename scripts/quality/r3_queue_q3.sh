#!/bin/bash
cd /workspace/dub/dubai; V=/workspace/dub/artifacts/r3d/video
while pgrep -f "r3_video.sh r2m 03" >/dev/null; do sleep 20; done; echo "R2M_CANON_DONE $(date +%T)"
( for id in 04 03 05 01 02; do [ -f $V/${id}_r2m_fix30.manifest.json ] && [ $id != 04 ] && continue; bash scripts/quality/r3_fix30.sh $V/${id}_r2m.manifest.json; done; echo "FIX30_CANON_DONE $(date +%T)" ) &
CI=/workspace/dub/handoff/in/C
until [ -f $CI/audio-r2/READY ]; do sleep 30; done; echo "C_AUDIO_READY $(date +%T)"; ls $CI/audio-r2 | head
for id in 03 04 05 01 02; do
  d=$CI/audio-r2/$id; [ -d $d ] || d=$CI/audio-r2-02/$id; [ -d $d ] || d=$(ls -d $CI/audio-r2-02 2>/dev/null)
  [ -f $d/dubbed_16k.wav ] || { echo "WARN no C audio for $id ($d)"; continue; }
  ARM_FLAGS="--anchor-filter flow:4 --anchor-round off --affine-precision fp32 --mask-track head --dump-ls-raw on" DUBBED_DIR_TPL=$d bash scripts/quality/r3_video.sh r2mC $id
  bash scripts/quality/r3_fix30.sh $V/${id}_r2mC.manifest.json &
done
wait; echo "R2MC_DONE $(date +%T)"
TAG=r3e2e bash scripts/r3_e2e.sh 04 /workspace/dub/artifacts/r3d/e2e; echo E2E04_DONE
