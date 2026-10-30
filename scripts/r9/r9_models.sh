#!/bin/bash
# r9: the pieces setup_r6.sh does not fetch (pre-approved in brief_r9_e2e.md): Qwen 14B Q5_K_M, CodeFormer (NC, DUB_ALLOW_NC), Real-ESRGAN x2,
# htdemucs (NC, quarantine torch home), Spleeter venv + weights, noisereduce.
set -u
R=/workspace/dub/dubai; cd $R; . scripts/env.sh >/dev/null; df0=$(df -B1 /workspace | tail -1 | awk '{print $3}')
dl(){ [ -s "$2" ] || { mkdir -p "$(dirname "$2")"; curl -fsSL --retry 3 -o "$2.part" "$1" && mv "$2.part" "$2"; }; echo "ok $(du -h "$2" | cut -f1) $2"; }
uv pip install --python $DUB_ENV/bin/python noisereduce 2>&1 | tail -1
python - <<'PY'
from huggingface_hub import snapshot_download
p = snapshot_download("Qwen/Qwen2.5-14B-Instruct-GGUF", allow_patterns=["*q5_k_m*.gguf"], local_dir="/workspace/dub/dubai/models/qwen2.5-14b-instruct-gguf")
import glob, os; print("qwen14b", [(os.path.basename(f), os.path.getsize(f)) for f in sorted(glob.glob(p + "/*.gguf"))])
PY
dl https://github.com/sczhou/CodeFormer/releases/download/v0.1.0/codeformer.pth $R/third_party/CodeFormer/weights/CodeFormer/codeformer.pth
dl https://github.com/sczhou/CodeFormer/releases/download/v0.1.0/parsing_parsenet.pth $R/third_party/CodeFormer/weights/facelib/parsing_parsenet.pth
dl https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.1/RealESRGAN_x2plus.pth $R/models/realesrgan/RealESRGAN_x2plus.pth
echo "49fafd45f8fd7aa8d31ab2a22d14d91b536c34494a5cfe31eb5d89c2fa266abb  $R/models/realesrgan/RealESRGAN_x2plus.pth" | sha256sum -c -
uv pip install --python $DUB_ENV/bin/python --target /workspace/dub/sep_pylib --no-deps demucs==4.0.1 dora-search openunmix julius lameenc treetable submitit retrying 2>&1 | tail -1
mkdir -p /workspace/dub/quarantine_nc/torch
TORCH_HOME=/workspace/dub/quarantine_nc/torch PYTHONPATH=/workspace/dub/sep_pylib:${PYTHONPATH:-} python -c "from demucs.pretrained import get_model; m = get_model('htdemucs'); print('htdemucs ok', type(m).__name__)" 2>&1 | tail -1
ls -la /workspace/dub/quarantine_nc/torch/hub/checkpoints 2>/dev/null
bash scripts/setup_spleeter.sh 2>&1 | tail -3
df1=$(df -B1 /workspace | tail -1 | awk '{print $3}'); echo "DISK_ADDED $(( (df1 - df0) / 1000000 )) MB; free $(df -h /workspace | tail -1 | awk '{print $4}')"
echo R9_MODELS_DONE
