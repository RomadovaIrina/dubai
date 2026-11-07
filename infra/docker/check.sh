#!/bin/bash
# DabAI image (r15 T): self-test of a worker (inside the container: `docker run --gpus all -v dabai-models:/workspace/dub/dubai/models <image> check`).
#     check.sh [--deep] [--weights-only]       exit 0 = ready for jobs
set -uo pipefail
R=${DUB_ROOT:-/workspace/dub/dubai}; M=$R/models; V=${VENV_ROOT:-/venv}; F=0; DEEP=""; WO=0
for a in "$@"; do case $a in --deep) DEEP=--deep;; --weights-only) WO=1;; esac; done
ok(){ echo "OK   $*"; }; bad(){ echo "FAIL $*"; F=1; }
${V}/dabai/bin/python $R/infra/docker/models_lock.py check --models $M $DEEP && ok "weights (models.lock.json)" || bad "weights: see above (run fetch-models)"
[ $WO = 1 ] && exit $F
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader 2>/dev/null | sed 's/^/OK   gpu: /' || bad "no GPU visible (--gpus all? NVIDIA container toolkit?)"
${V}/dabai/bin/python - <<'PY' && ok "torch runs on the GPU" || bad "torch on the GPU"
import torch; assert torch.cuda.is_available(), "cuda not available"; c = torch.cuda.get_device_capability(0)
x = torch.randn(1024, 1024, device="cuda", dtype=torch.float16); y = (x @ x).float().sum().item()   # a real sm_120 kernel, not only the driver
print(f"     torch {torch.__version__} cuda {torch.version.cuda} capability {c} {torch.cuda.get_device_name(0)} vram {torch.cuda.get_device_properties(0).total_memory / 2**30:.1f} GiB")
assert torch.cuda.get_device_properties(0).total_memory >= 30 * 2**30, "the pipeline needs a 32 GB card (audio stage peak ~30 GB)"
PY
PYTHONPATH=$R/third_party/llama_cuda ${V}/dabai/bin/python -c "import llama_cpp, sys; sys.exit(0 if llama_cpp.llama_supports_gpu_offload() else 1)" && ok "llama.cpp CUDA build (Qwen 14B on the GPU)" || bad "llama_cuda"
${V}/mediapipe/bin/python -c "import mediapipe" && ok "mediapipe venv" || bad "mediapipe venv"
${V}/main/bin/python -c "import pynvml; pynvml.nvmlInit()" && ok "NVML (GPU accounting)" || bad "NVML"
# (outputs captured first: with pipefail, "cmd | grep -q" fails when grep exits early and cmd dies of SIGPIPE)
ENC=$(ffmpeg -hide_banner -encoders 2>/dev/null); FLT=$(ffmpeg -hide_banner -filters 2>/dev/null); FON=$(fc-list 2>/dev/null)
grep -q libx264 <<< "$ENC" && grep -q " subtitles " <<< "$FLT" && ok "ffmpeg libx264 + libass" || bad "ffmpeg libx264 / libass"
grep -qi "DejaVu Sans" <<< "$FON" && ok "subtitle fonts" || bad "DejaVu Sans font"
echo "     profile: DUB_PROFILE=${DUB_PROFILE:-unset} (DUB_ALLOW_NC=${DUB_ALLOW_NC:-?} DUB_ALLOW_NC_BED=${DUB_ALLOW_NC_BED:-?})"
if [ "${DUB_ALLOW_NC:-0}" != 1 ]; then
  [ -e $R/third_party/CodeFormer ] && bad "CodeFormer present while DUB_ALLOW_NC=0" || ok "clean profile: no CodeFormer"
  ${V}/dabai/bin/python -c "import sys; sys.path.insert(0, '$R/scripts/pilot'); import clean_face_stack as c; c.install_tripwires(); c.assert_clean_tree()" > /dev/null && ok "no banned weight files (clean_face_stack)" || bad "banned weight files present"
else
  [ -s $R/third_party/CodeFormer/weights/CodeFormer/codeformer.pth ] && ok "current profile: CodeFormer (NC) present" || bad "current profile: no CodeFormer weights"
  [ -s /workspace/dub/quarantine_nc/torch/hub/checkpoints/955717e8-8726e21a.th ] && [ -d /workspace/dub/sep_pylib/demucs ] && ok "current profile: htdemucs (NC) present" || bad "current profile: no htdemucs"
fi
[ -s "${DABAI_STUDENT_LORA:-/nonexistent}" ] && ok "student LoRA (mode fast)" || echo "WARN no student LoRA: mode fast unavailable"
df -BG --output=avail /data 2>/dev/null | tail -1 | awk '{g=$1+0; if (g < 60) print "WARN /data has only " g " GB free (a job needs >= 10 GB per video minute while it runs: 5 min >= 60 GB, 15 min ~150 GB)"; else print "OK   /data " g " GB free"}'
[ $F = 0 ] && echo "CHECK_OK" || echo "CHECK_FAILED"; exit $F
