#!/bin/bash
# r3 D queue v5 (log name kept: q4.log, the scorers yield on its markers): r2m batch -> GPU audit (alone) -> FIX30 (CPU) || r2m on C audio-r2 (GPU) -> E2E timing
cd /workspace/dub/dubai; V=/workspace/dub/artifacts/r3d/video; CI=/workspace/dub/handoff/in/C
while pgrep -f "r3_video[.]sh r2m 03" >/dev/null; do sleep 20; done; echo "R2M_CANON_DONE $(date +%T)"
PYTHONPATH=third_party/llama_cuda /venv/dabai/bin/python -c "import llama_cpp;print('LLAMA_CUDA', llama_cpp.__version__, llama_cpp.llama_supports_gpu_offload())"
while pgrep -f "r3_scorecard[.]py score" >/dev/null; do sleep 15; done
bash scripts/quality/r3_gpu_audit.sh gpu 04 03; echo "AUDIT_GPU_DONE $(date +%T)"
bash scripts/quality/r3_gpu_audit.sh cpuqwen 04; echo "AUDIT_CPUQWEN_DONE $(date +%T)"
bash scripts/quality/r3_gpu_audit.sh resident 04 03; echo "AUDIT_RESIDENT_DONE $(date +%T)"
( for id in 04 03 05 01 02; do [ -f $V/${id}_r2m_fix30.mp4 ] && [ $id != 04 ] && continue; bash scripts/quality/r3_fix30.sh $V/${id}_r2m.manifest.json; done; echo "FIX30_CANON_DONE $(date +%T)" ) &
for id in 03 04 05 01 02; do
  d=$CI/audio-r2/$id; [ -d $d ] || d=$CI/audio-r2-02/$id
  [ -f $d/dubbed_16k.wav ] || { echo "WARN no C audio for $id ($d)"; continue; }
  ARM_FLAGS="--anchor-filter flow:4 --anchor-round off --affine-precision fp32 --mask-track head --dump-ls-raw on" DUBBED_DIR_TPL=$d MANIFEST_TPL=$d/${id}_r2.manifest.json \
    bash scripts/quality/r3_video.sh r2mC $id
  bash scripts/quality/r3_fix30.sh $V/${id}_r2mC.manifest.json &
done
wait; echo "R2MC_DONE $(date +%T)"
TAG=r3e2e bash scripts/r3_e2e.sh 04 /workspace/dub/artifacts/r3d/e2e; echo E2E04_DONE
