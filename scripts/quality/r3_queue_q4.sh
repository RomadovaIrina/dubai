#!/bin/bash
# r3 D queue v4: r2m batch -> GPU audit (alone) -> FIX30 (CPU) || r2m on C audio (GPU) -> E2E timing
cd /workspace/dub/dubai; V=/workspace/dub/artifacts/r3d/video
while pgrep -f "r3_video[.]sh r2m 03" >/dev/null; do sleep 20; done; echo "R2M_CANON_DONE $(date +%T)"
until [ -f third_party/llama_cuda/llama_cpp/__init__.py ] && ! pgrep -f "setup_llama_cuda[.]sh" >/dev/null; do sleep 20; done
PYTHONPATH=third_party/llama_cuda /venv/dabai/bin/python -c "import llama_cpp;print('LLAMA_CUDA', llama_cpp.__version__, llama_cpp.llama_supports_gpu_offload())"
while pgrep -f "r3_scorecard[.]py score" >/dev/null; do sleep 15; done
bash scripts/quality/r3_gpu_audit.sh gpu 04 03; echo "AUDIT_GPU_DONE $(date +%T)"
bash scripts/quality/r3_gpu_audit.sh cpuqwen 04; echo "AUDIT_CPUQWEN_DONE $(date +%T)"
bash scripts/quality/r3_gpu_audit.sh resident 04 03; echo "AUDIT_RESIDENT_DONE $(date +%T)"
( for id in 04 03 05 01 02; do bash scripts/quality/r3_fix30.sh $V/${id}_r2m.manifest.json; done; echo "FIX30_CANON_DONE $(date +%T)" ) &
CI=/workspace/dub/handoff/in/C
until [ -f $CI/audio-r2/READY ]; do sleep 30; done; echo "C_AUDIO_READY $(date +%T)"; ls $CI/audio-r2 | head
for id in 03 04 05 01 02; do
  d=$CI/audio-r2/$id; [ -d $d ] || d=$CI/audio-r2-02/$id; [ -d $d ] || d=$CI/audio-r2-02
  [ -f $d/dubbed_16k.wav ] || { echo "WARN no C audio for $id ($d)"; continue; }
  ARM_FLAGS="--anchor-filter flow:4 --anchor-round off --affine-precision fp32 --mask-track head --dump-ls-raw on" DUBBED_DIR_TPL=$d bash scripts/quality/r3_video.sh r2mC $id
  bash scripts/quality/r3_fix30.sh $V/${id}_r2mC.manifest.json &
done
wait; echo "R2MC_DONE $(date +%T)"
TAG=r3e2e bash scripts/r3_e2e.sh 04 /workspace/dub/artifacts/r3d/e2e; echo E2E04_DONE
