#!/bin/bash
# r15-K: KeySync env (r13 patch) + weights
set -e; cd /workspace/dub/keysync
[ -d code ] || git clone -q https://github.com/antonibigata/keysync code
cd code && git checkout -q 5e2503b && (git apply --check ../keysync_5e2503b_sm120_r15.patch 2>/dev/null && git apply ../keysync_5e2503b_sm120_r15.patch && echo PATCH_OK || echo PATCH_ALREADY_OR_FAIL)
for n in 6 8; do for c in keyframe interpolation; do sed "s/num_steps: 10/num_steps: $n/" scripts/sampling/configs/$c.yaml > scripts/sampling/configs/${c}_s$n.yaml; done; done
cd ..; grep -v -i -E "^(torch==|torchvision|torchaudio|nvidia-|triton|xformers|deepspeed|bitsandbytes|dlib|face-recognition|face_recognition|insightface|onnx)" code/requirements.txt > req_ks.txt
[ -d venv ] || uv venv -q -p 3.11 venv; . venv/bin/activate
uv pip install -q torch==2.7.1 torchvision==0.22.1 torchaudio==2.7.1 --index-url https://download.pytorch.org/whl/cu128
uv pip install -q --no-deps -r req_ks.txt
uv pip install -q "omegaconf==2.3.0" "antlr4-python3-runtime==4.9.3" face-alignment==1.4.1 torchao==0.11.0 hf_xet
echo VENV_DONE
python dl.py
cd code && python -c "import sys; sys.path.insert(0,\".\"); import scripts.sampling.dubbing_pipeline_raw" && echo KS_SETUP_DONE
