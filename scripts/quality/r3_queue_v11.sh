#!/bin/bash
# r3 D queue v11 (coordinator 19.10): r3cand on C audio-r3 (A R3 generator + B fps master, B final composite) -> same-audio A/B arms on audio-r3
#   (r3F generator, clean0; 25 fps, SyncNet comparison) -> audio-r3b remux of the r3cand finals when READY
cd /workspace/dub/dubai; V=/workspace/dub/artifacts/r3d/video; CI=/workspace/dub/handoff/in/C; A3=$CI/audio-r3
while pgrep -f "r3_gpu_audit[.]sh" >/dev/null; do sleep 20; done
for id in 03 04 05 01 02; do
  d=$A3/$id; DUBBED_DIR_TPL=$d MANIFEST_TPL=$d/${id}_r3.manifest.json bash scripts/quality/r3_video.sh r3cand $id
  FIX30_MODE=r3cand FIX30_THREADS=8 bash scripts/quality/r3_fix30.sh $V/${id}_r3cand.manifest.json &
done
echo "R3CAND_RENDERS_DONE $(date +%T)"
for arm in r3Fa3 clean0a3; do for id in 03 05 04 01; do
  d=$A3/$id; ARM_FLAGS=" " DUBBED_DIR_TPL=$d MANIFEST_TPL=$d/${id}_r3.manifest.json bash scripts/quality/r3_video.sh $arm $id
done; done
wait; echo "AB_A3_DONE $(date +%T)"
until [ -f $CI/audio-r3b/READY ]; do sleep 30; done; echo "C_AUDIO_R3B_READY $(date +%T)"
bash scripts/quality/r3_remux_r3b.sh; echo "R3B_REMUX_DONE $(date +%T)"
