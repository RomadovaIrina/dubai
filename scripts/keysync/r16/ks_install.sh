#!/bin/bash
# r16 T: install the KeySync-fast kit into KS_ROOT (default /workspace/dub/keysync). Idempotent; used on research servers AND in the Docker image.
#   = r15 K k_setup.sh (keysync 5e2503b + the sm120 r15 patch, torch 2.7.1 cu128 venv, HF toninio19/keysync weights) + TensorRT 10.16 + the kit scripts.
#   KS_NO_WEIGHTS=1: code + venv only (the Docker build; the weights live on the models volume, scripts/keysync/r16/ks_engines.sh fetches them on first start).
set -e
R=$(cd "$(dirname "$0")/../../.." && pwd); KS=${KS_ROOT:-/workspace/dub/keysync}; mkdir -p $KS; cd $KS
# the kit: the r12-r15 KeySync scripts the face stage calls (ks_prep, ks_fast, ks_trt, ks_student, ks_plain, k_faces2, ...)
cp $R/scripts/keysync/*.py $R/scripts/keysync/*.patch $KS/; cp $R/scripts/keysync/r15/*.py $KS/; cp -r $R/scripts/keysync/configs $KS/ 2>/dev/null || true
[ -d code/.git ] || git clone -q https://github.com/antonibigata/keysync code
cd code; git checkout -q 5e2503b
if git apply --check ../keysync_5e2503b_sm120_r15.patch 2>/dev/null; then git apply ../keysync_5e2503b_sm120_r15.patch && echo PATCH_OK
elif git apply -R --check ../keysync_5e2503b_sm120_r15.patch 2>/dev/null; then echo PATCH_ALREADY
else echo "PATCH_FAIL (another patch applied?)"; git status --short | head; exit 1; fi
for n in 6 8; do for c in keyframe interpolation; do sed "s/num_steps: 10/num_steps: $n/" scripts/sampling/configs/$c.yaml > scripts/sampling/configs/${c}_s$n.yaml; done; done
cd ..; grep -v -i -E "^(torch==|torchvision|torchaudio|nvidia-|triton|xformers|deepspeed|bitsandbytes|dlib|face-recognition|face_recognition|insightface|onnx)" code/requirements.txt > req_ks.txt
[ -x venv/bin/python ] || uv venv -q -p 3.11 venv
. venv/bin/activate
python -c "import torch, tensorrt" 2>/dev/null && [ "$(python -c 'import torch; print(torch.__version__)')" = 2.7.1+cu128 ] || {
  uv pip install -q torch==2.7.1 torchvision==0.22.1 torchaudio==2.7.1 --index-url https://download.pytorch.org/whl/cu128
  uv pip install -q --no-deps -r req_ks.txt
  uv pip install -q "omegaconf==2.3.0" "antlr4-python3-runtime==4.9.3" face-alignment==1.4.1 torchao==0.11.0 hf_xet tensorrt-cu12==10.16.1.11 onnx==1.18.0; }
echo VENV_DONE
if [ "${KS_NO_WEIGHTS:-0}" != 1 ]; then python dl.py; fi
cd code && python -c "import sys; sys.path.insert(0, '.'); import scripts.sampling.dubbing_pipeline_raw, tensorrt; print('trt', tensorrt.__version__)" && echo KS_INSTALL_DONE
