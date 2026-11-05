#!/bin/bash
# r15 W0: TBDub (TaoLiveAIGC/TBDub, Apache-2.0 code + weights) for the bake-off: code clone, the STUDENT checkpoint + shared files (~15.3 GB:
# tbdub_student 12.59 GB sha256-checked, Wan2.2_VAE 1.41 GB from KlingTeam/X-Dub (apache-2.0), facebook/hubert-large-ll60k 1.26 GB (apache-2.0),
# MediaPipe face_landmarker 4 MB (Apache-2.0)), and its own venv (diffusers 0.36 / transformers 5.1 clash with the dabai env) that reuses the
# dabai env's torch 2.7.1+cu128 (Blackwell) through a .pth fallback. The teacher (another 12.59 GB) is NOT fetched.
#   bash scripts/r15/tbdub_setup.sh            -> /workspace/dub/tbdub/{TBDub, ckpt, venv}
set -e
T=/workspace/dub/tbdub; D=$T/ckpt; mkdir -p $T $D/hubert-large-ll60k; cd $T
[ -d TBDub ] || git clone -q https://github.com/TaoLiveAIGC/TBDub.git   # r15: 19e18e4
(cd /workspace/dub/dubai; . scripts/env.sh >/dev/null 2>&1; python - <<'PY'
from huggingface_hub import hf_hub_download
D = "/workspace/dub/tbdub/ckpt"
for f in ["config.json", "null_prompt_emb.pt", "tbdub_student.safetensors"]: hf_hub_download("TaoLiveAIGC/TBDub", f, local_dir=D)
hf_hub_download("KlingTeam/X-Dub", "Wan2.2_VAE.safetensors", local_dir=D)
for f in ["config.json", "preprocessor_config.json", "pytorch_model.bin"]: hf_hub_download("facebook/hubert-large-ll60k", f, local_dir=D + "/hubert-large-ll60k")
PY
)
[ -s $D/face_landmarker.task ] || curl -fsSL https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task -o $D/face_landmarker.task
(cd $D; echo '64184e229b263107bc2b804c6625db1341ff2bb731874b0bcc2fe6544e0bc9ff  face_landmarker.task' | sha256sum -c -
 echo '4e86900f9744b3650bfe58451ae89b3afba758233a3edf5947516268c1ea11e6  tbdub_student.safetensors' | sha256sum -c -)
if [ ! -x $T/venv/bin/python ]; then
  SP=$(/venv/dabai/bin/python -c "import site;print(site.getsitepackages()[0])")
  uv venv --python /venv/dabai/bin/python $T/venv
  VSP=$($T/venv/bin/python -c "import site;print(site.getsitepackages()[0])"); echo "$SP" > "$VSP/zz_dabai_fallback.pth"   # torch / torchvision from dabai
  VIRTUAL_ENV=$T/venv uv pip install -r TBDub/requirements.txt hf_xet
fi
$T/venv/bin/python -c "import torch, diffusers, transformers, mediapipe, cv2; print('TBDub env OK', torch.__version__, diffusers.__version__, transformers.__version__, torch.cuda.is_available())"
