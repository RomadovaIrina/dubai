#!/bin/bash
# r3 D queue v10: (1) B's final composite (r3cand mode) on the r3F raw layers = r3cand on audio-r2 (CPU, 6 threads)
#   (2) when C's audio-r3 is READY: r3cand on audio-r3 (A r2best generator + B fps master; B final composite) for 03 04 05 01 02
cd /workspace/dub/dubai; V=/workspace/dub/artifacts/r3d/video; CI=/workspace/dub/handoff/in/C
( for id in 03 04 05 01 02; do [ -f $V/${id}_r3F_r3cand.mp4 ] || FIX30_MODE=r3cand FIX30_THREADS=6 bash scripts/quality/r3_fix30.sh $V/${id}_r3F.manifest.json r3cand; done; echo "R3CAND_AUDIO_R2_DONE $(date +%T)" ) &
until [ -f $CI/audio-r3/READY ]; do sleep 30; done; echo "C_AUDIO_R3_READY $(date +%T)"; ls $CI/audio-r3
while pgrep -f "r3_gpu_audit[.]sh" >/dev/null; do sleep 20; done
for id in 03 04 05 01 02; do
  d=$CI/audio-r3/$id; [ -d $d ] || d=$(ls -d $CI/audio-r3*/$id 2>/dev/null | head -1)
  [ -n "$d" ] && [ -f $d/dubbed_16k.wav ] || { echo "WARN no audio-r3 for $id"; continue; }
  M=$(ls $d/${id}_*.manifest.json 2>/dev/null | head -1); [ -n "$M" ] || { echo "WARN no manifest in $d"; continue; }
  DUBBED_DIR_TPL=$d MANIFEST_TPL=$M bash scripts/quality/r3_video.sh r3cand $id
  FIX30_MODE=r3cand FIX30_THREADS=8 bash scripts/quality/r3_fix30.sh $V/${id}_r3cand.manifest.json &
done
wait; echo "R3CAND_AUDIO_R3_DONE $(date +%T)"
