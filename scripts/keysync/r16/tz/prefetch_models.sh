#!/bin/bash
# r16 TZ setup part B on r16-q: every weight the KeySync e2e path needs, fetched with parallel connections (hf_transfer / hf_xet in a tiny
# side venv, huggingface_hub pinned to the pipeline's 0.30.2 so download_models.sh later sees complete local_dir files and skips them).
# Same files / patterns as scripts/download_models.sh (minus the unused Qwen 7B) + the Qwen 14B Q5_K_M + KeySync + the hub-cache files
# weights_check.py wants. NO NC weights: CodeFormer / ParseNet / htdemucs (r9_models.sh) are never fetched here. The token is never printed.
set -u
R=/workspace/dub/dubai-tz; M=$R/models; K=/workspace/dub/keysync; E=/workspace/dub/dlenv; mkdir -p $M $K
export HF_HOME=$M/hf HF_TOKEN_PATH=/root/.cache/huggingface/token HF_HUB_DISABLE_TELEMETRY=1 HF_HUB_ENABLE_HF_TRANSFER=1 HF_XET_HIGH_PERFORMANCE=1
t0=$(date +%s); df0=$(df -B1 /workspace | tail -1 | awk '{print $3}')
[ -x $E/bin/python ] || { uv venv -q -p 3.12 $E && uv pip install -q --python $E/bin/python huggingface_hub==0.30.2 hf_transfer hf_xet; } || { echo PREFETCH_VENV_FAILED; exit 1; }
[ -d $K/code ] || git clone -q https://github.com/antonibigata/keysync $K/code   # tz_setup_a.sh then only checks out 5e2503b + patches
$E/bin/python - <<'PY'
import os, time
from huggingface_hub import snapshot_download, hf_hub_download
M = "/workspace/dub/dubai-tz/models"; K = "/workspace/dub/keysync/code/pretrained_models"
jobs = [  # (repo, local_dir or None = the hub cache, allow_patterns)
    ("toninio19/keysync", K, ["checkpoints/WavLM-Base+.pt", "checkpoints/keyframe_dub.pt", "checkpoints/interpolation_dub.pt"]),
    ("Qwen/Qwen2.5-14B-Instruct-GGUF", f"{M}/qwen2.5-14b-instruct-gguf", ["*q5_k_m*.gguf"]),
    ("Systran/faster-whisper-large-v3", f"{M}/whisper-large-v3", None),
    ("ResembleAI/chatterbox", f"{M}/chatterbox", ["ve.pt", "s3gen.pt", "conds.pt", "t3_mtl23ls_v3.safetensors", "grapheme_mtl_merged_expanded_v1.json", "Cangjie5_TC.json", "*.json"]),
    ("ByteDance/LatentSync-1.6", f"{M}/latentsync", ["latentsync_unet.pt", "whisper/tiny.pt", "auxiliary/syncnet_v2.model", "auxiliary/sfd_face.pth"]),
    ("stabilityai/sd-vae-ft-mse", f"{M}/sd-vae-ft-mse", ["*.json", "*.safetensors"]),
    ("facebook/wav2vec2-base-960h", f"{M}/wav2vec2-base-960h", ["*.json", "model.safetensors"]),
    ("facebook/hubert-base-ls960", None, ["*.json", "pytorch_model.bin"]),
    ("stabilityai/stable-video-diffusion-img2vid", None, ["vae/config.json", "vae/diffusion_pytorch_model.fp16.safetensors"]),
]
bad = 0
for repo, d, pat in jobs:
    t = time.time()
    for k in range(3):
        try:
            p = snapshot_download(repo_id=repo, local_dir=d, allow_patterns=pat, max_workers=8)
            n = sum(os.path.getsize(os.path.join(a, f)) for a, _, fs in os.walk(p) for f in fs if ".cache" not in a)
            print(f"ok {repo} {n / 1e9:.2f} GB {time.time() - t:.0f} s -> {p}", flush=True); break
        except Exception as e:
            print(f"retry {k} {repo}: {type(e).__name__}: {str(e)[:200]}", flush=True); time.sleep(5)
    else:
        bad += 1; print(f"FAILED {repo}", flush=True)
print("PREFETCH_PY_DONE bad", bad, flush=True)
PY
mkdir -p $M/realesrgan; f=$M/realesrgan/RealESRGAN_x2plus.pth   # BSD-3 (the LatentSync fallback's weights_check entry)
[ -s $f ] || aria2c -q -x 8 -s 8 -d $M/realesrgan -o RealESRGAN_x2plus.pth https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.1/RealESRGAN_x2plus.pth
echo "49fafd45f8fd7aa8d31ab2a22d14d91b536c34494a5cfe31eb5d89c2fa266abb  $f" | sha256sum -c - | tail -1
df1=$(df -B1 /workspace | tail -1 | awk '{print $3}'); echo "PREFETCH_DONE $(( $(date +%s) - t0 )) s, disk +$(( (df1 - df0) / 1000000000 )) GB"
