#!/bin/bash
# r3 D queue v8 (log q4.log; scorers yield between R2M_CANON_DONE and AUDIT_RESIDENT_DONE). [v7's FIX30 subshell keeps running on its own.]
#   r2mC 03 (running) -> r3F on C audio-r2 01-05 (+ FIX30 r3F behind each) -> GPU audit: (a) gpu 03 04, cpuqwen 04, (b) resaudio 04 03, (all) resident 04 03
#   -> E2E timing
cd /workspace/dub/dubai; V=/workspace/dub/artifacts/r3d/video; CI=/workspace/dub/handoff/in/C
while pgrep -f "r3_video[.]sh r2mC 03" >/dev/null; do sleep 10; done
for id in 03 04 05 01 02; do
  d=$CI/audio-r2/$id; [ -d $d ] || d=$CI/audio-r2-02/$id
  [ -f $d/dubbed_16k.wav ] || { echo "WARN no C audio for $id ($d)"; continue; }
  [ -f $V/${id}_r3F.manifest.json ] || DUBBED_DIR_TPL=$d MANIFEST_TPL=$d/${id}_r2.manifest.json bash scripts/quality/r3_video.sh r3F $id
  FIX30_MODE=r3F FIX30_THREADS=8 bash scripts/quality/r3_fix30.sh $V/${id}_r3F.manifest.json &
done
wait; echo "R3F_DONE $(date +%T)"
while pgrep -f "r3_fix30[.]sh" >/dev/null; do sleep 15; done
echo "R2M_CANON_DONE $(date +%T)"
while pgrep -f "r3_scorecard[.]py score" >/dev/null; do sleep 15; done
bash scripts/quality/r3_gpu_audit.sh gpu 03 04; echo "AUDIT_GPU_DONE $(date +%T)"
bash scripts/quality/r3_gpu_audit.sh cpuqwen 04; echo "AUDIT_CPUQWEN_DONE $(date +%T)"
bash scripts/quality/r3_gpu_audit.sh resaudio 04 03; echo "AUDIT_RESAUDIO_DONE $(date +%T)"
bash scripts/quality/r3_gpu_audit.sh resident 04 03; echo "AUDIT_RESIDENT_DONE $(date +%T)"
TAG=r3e2e bash scripts/r3_e2e.sh 04 /workspace/dub/artifacts/r3d/e2e; echo E2E04_DONE
