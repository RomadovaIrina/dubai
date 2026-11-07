#!/bin/bash
# DabAI image (r15 T): fill the MODELS VOLUME once (the image has no weights). Commercially clean set only; idempotent (re-run = resume).
#     docker run --rm --gpus all -e HF_TOKEN=<token with the pyannote 3.1 terms accepted> -v dabai-models:/workspace/dub/dubai/models \
#         [-v /path/with/student:/in:ro -e STUDENT_LORA_FILE=/in/latentsync_student_r2_s4_step01500_ema.pt] <image> fetch-models
# What lands on the volume (~24 GB; list + licences: infra/docker/models.lock.json, reports/experiments/r13-spec/IMAGE/MODELS.md):
#   whisper-large-v3 (MIT) · Qwen2.5-14B-Instruct Q5_K_M (Apache-2.0) · Chatterbox Multilingual v3 (MIT) · LatentSync 1.6 (OpenRAIL++, approved)
#   sd-vae-ft-mse (MIT) · wav2vec2-base-960h (Apache-2.0) · MediaPipe Face Landmarker (Apache-2.0) · Real-ESRGAN x2plus (BSD-3) · Spleeter 2stems (MIT)
#   pyannote speaker-diarization-3.1 (MIT, gated) -> cache/pyannote · Chatterbox's pkuseg model -> cache/pkuseg · the student LoRA (ours) -> student/
# NEVER fetched: CodeFormer / ParseNet, htdemucs, InsightFace buffalo_l, RetinaFace R50 (non-commercial); clean_face_stack tripwires stay armed.
set -euo pipefail
R=${DUB_ROOT:-/workspace/dub/dubai}; M=$R/models; PY=${DUB_ENV:-/venv/dabai}/bin/python; mkdir -p $M/cache
unset HF_HUB_OFFLINE TRANSFORMERS_OFFLINE; export HF_HOME=$M/hf HF_HUB_DISABLE_TELEMETRY=1 PYANNOTE_CACHE=$M/cache/pyannote
log(){ echo "[$(date +%T)] == $*"; }
: "${HF_TOKEN:?HF_TOKEN is needed once (gated pyannote/speaker-diarization-3.1); it is not stored on the volume}"
dl(){ [ -s "$2" ] || { mkdir -p "$(dirname "$2")"; curl -fsSL --retry 3 -o "$2.part" "$1" && mv "$2.part" "$2"; }; }

log "1/7 whisper large-v3, Chatterbox v3, LatentSync 1.6, sd-vae, wav2vec2, MediaPipe task (scripts/download_models.sh, Qwen 7B skipped)"
DUB_SKIP_QWEN7B=1 PY=$PY bash $R/scripts/download_models.sh
log "2/7 Qwen2.5-14B-Instruct GGUF Q5_K_M (10.5 GB, revision pinned)"
$PY - <<'PY'
from huggingface_hub import snapshot_download
snapshot_download("Qwen/Qwen2.5-14B-Instruct-GGUF", revision="b466e1f8c07172155743e8e1307507d8a4f91fbd", allow_patterns=["*q5_k_m*.gguf"],
                  local_dir="/workspace/dub/dubai/models/qwen2.5-14b-instruct-gguf"); print("qwen14b ok")
PY
log "3/7 Real-ESRGAN x2plus (sha256 pinned)"
dl https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.1/RealESRGAN_x2plus.pth $M/realesrgan/RealESRGAN_x2plus.pth
echo "49fafd45f8fd7aa8d31ab2a22d14d91b536c34494a5cfe31eb5d89c2fa266abb  $M/realesrgan/RealESRGAN_x2plus.pth" | sha256sum -c -
log "4/7 Spleeter 2stems (MIT) -> the npz of the torch port (a throw-away TF 2.9 CPU venv converts it, then is deleted)"
if [ ! -s $M/spleeter/2stems_tf_vars.npz ]; then
  T=$(mktemp -d); DUB_SPLEETER_VENV=$T/venv bash $R/scripts/setup_spleeter.sh; rm -rf $T
fi
log "5/7 pyannote speaker-diarization-3.1 (gated) -> $PYANNOTE_CACHE"
TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1 $PY -c "
import os
from pyannote.audio import Pipeline
p = Pipeline.from_pretrained('pyannote/speaker-diarization-3.1', use_auth_token=os.environ['HF_TOKEN']); assert p is not None, 'accept the pyannote 3.1 terms for this token'
print('pyannote ok')"
log "6/7 pkuseg (Chatterbox's Chinese segmenter) -> cache/pkuseg"
$PY -c "import pkuseg; pkuseg.pkuseg(); print('pkuseg ok')" || $PY -c "import spacy_pkuseg as p; p.pkuseg(); print('pkuseg ok')" || echo "pkuseg prefetch failed (only Chinese needs it)"
[ -d /root/.pkuseg ] && [ ! -L /root/.pkuseg ] && { rm -rf $M/cache/pkuseg; cp -r /root/.pkuseg $M/cache/pkuseg; }
log "7/7 the student LoRA (mode fast; our own weights, not on a public hub)"
S=$M/student/latentsync_student_r2_s4_step01500_ema.pt; mkdir -p $M/student
if [ -s $S ]; then echo "have $S"
elif [ -n "${STUDENT_LORA_FILE:-}" ]; then cp "$STUDENT_LORA_FILE" $S
elif [ -n "${STUDENT_LORA_URL:-}" ]; then dl "$STUDENT_LORA_URL" $S
else echo "WARNING: no student LoRA (STUDENT_LORA_FILE / STUDENT_LORA_URL): mode fast is unavailable until $S exists"; fi
if [ "${DUB_PROFILE:-}" = current ]; then   # the NC set, ONLY for the explicit current profile, onto its OWN volume (never the models volume)
  N=${DABAI_NC_DIR:-/opt/dabai-nc}; mkdir -p $N; [ -w $N ] || { echo "mount a writable NC volume at $N" >&2; exit 64; }
  log "NC set for DUB_PROFILE=current (CodeFormer S-Lab NC + htdemucs weights: a deliberate licence deviation, user-accepted 29.10) -> $N"
  CF=b33cc7d639d6545bfcccc7e0bc6ae51f24e79c2b
  if [ ! -f $N/CodeFormer/basicsr/version.py ]; then
    T=$(mktemp -d); curl -fsSL --retry 3 -o $T/cf.tar.gz https://github.com/sczhou/CodeFormer/archive/$CF.tar.gz && tar -xzf $T/cf.tar.gz -C $T
    rm -rf $N/CodeFormer; mv $T/CodeFormer-$CF $N/CodeFormer; rm -rf $T
    printf "# GENERATED VERSION FILE (fetch_models.sh)\n__version__ = '1.3.2'\n__gitsha__ = '%s'\nversion_info = (1, 3, 2)\n" $CF > $N/CodeFormer/basicsr/version.py
  fi
  dl https://github.com/sczhou/CodeFormer/releases/download/v0.1.0/codeformer.pth $N/CodeFormer/weights/CodeFormer/codeformer.pth   # ParseNet: not used (--codeformer-parse off)
  echo "1009e537e0c2a07d4cabce6355f53cb66767cd4b4297ec7a4a64ca4b8a5684b7  $N/CodeFormer/weights/CodeFormer/codeformer.pth" | sha256sum -c -
  [ -d $N/sep_pylib/demucs ] || uv pip install --python $PY --target $N/sep_pylib --no-deps demucs==4.0.1 dora-search==0.1.13 julius==0.2.8 lameenc==1.9.0 \
      openunmix==1.3.0 retrying==1.4.2 submitit==1.5.4 treetable==0.2.6
  mkdir -p $N/quarantine_nc/torch; TORCH_HOME=$N/quarantine_nc/torch PYTHONPATH=$N/sep_pylib DUB_ALLOW_NC_BED=1 $PY -c "from demucs.pretrained import get_model; get_model('htdemucs'); print('htdemucs ok')"
  echo "8726e21a993978c7ba086d3872e7608d7d5bfca646ca4aca459ffda844faa8b4  $N/quarantine_nc/torch/hub/checkpoints/955717e8-8726e21a.th" | sha256sum -c -
fi
log "check"; bash $R/infra/docker/check.sh --weights-only
du -sh $M; echo FETCH_MODELS_DONE
