#!/bin/bash
# DabAI image (r15 T): fill the MODELS VOLUME once (the image has no weights). Commercially clean set only; idempotent (re-run = resume).
#     docker run --rm --gpus all -e HF_TOKEN=<token with the pyannote 3.1 terms accepted> -v dabai-models:/workspace/dub/dubai/models \
#         [-v /path/with/student:/in:ro -e STUDENT_LORA_FILE=/in/latentsync_student_r2_s4_step01500_ema.pt] <image> fetch-models
# What lands on the volume (~38 GB with KeySync, r16 TZ: models.lock.json 69 files 37.6 GB; list + licences: infra/docker/models.lock.json, reports/experiments/r13-spec/IMAGE/MODELS.md):
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
$PY - $M/qwen2.5-14b-instruct-gguf <<'PY'
import sys
from huggingface_hub import snapshot_download   # r16 TZ: local_dir from DUB_ROOT (was hard-coded to /workspace/dub/dubai: research boxes got no Qwen 14B)
snapshot_download("Qwen/Qwen2.5-14B-Instruct-GGUF", revision="b466e1f8c07172155743e8e1307507d8a4f91fbd", allow_patterns=["*q5_k_m*.gguf"],
                  local_dir=sys.argv[1]); print("qwen14b ok")
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
log "8/8 KeySync (r16 face): toninio19/keysync checkpoints (pinned revision) + the LCM student + the TensorRT capture (ours: KS_KIT_DIR)"
K=$M/keysync; mkdir -p $K/checkpoints $K/student $K/trt_cap $K/trt_cache $K/inductor $K/torch
$PY - $K <<'PY'
import os, sys, pathlib
from huggingface_hub import hf_hub_download   # r16 TZ: straight into $K/checkpoints (local_dir); the HF-cache copy doubled the 13 GB on the volume
for f in ["WavLM-Base+.pt", "keyframe_dub.pt", "interpolation_dub.pt"]:
    d = pathlib.Path(sys.argv[1]) / "checkpoints" / f
    if not d.exists(): hf_hub_download("toninio19/keysync", "checkpoints/" + f, revision=os.environ.get("KS_HF_REVISION", "main"), local_dir=sys.argv[1])
    print("keysync", f, d.stat().st_size)
PY
# KeySync's own runtime downloads (found by the r16 clean room, HF offline): HuBERT (audio features) + the SVD VAE (gated repo: the HF_TOKEN must
# have accepted its terms) into HF_HOME = $M/hf, revisions pinned; face_alignment's S3FD + 2DFAN4 into KS_TORCH_HOME (sha256 pinned)
$PY - <<'PY'
import pathlib
from huggingface_hub import snapshot_download
for repo, rev, pats in (("facebook/hubert-base-ls960", "dba3bb02fda4248b6e082697eee756de8fe8aa8a", ["config.json", "preprocessor_config.json", "pytorch_model.bin"]),
                        ("stabilityai/stable-video-diffusion-img2vid", "9cf024d5bfa8f56622af86c884f26a52f6676f2e", ["vae/config.json", "vae/diffusion_pytorch_model.fp16.safetensors"])):
    snap = pathlib.Path(snapshot_download(repo, revision=rev, allow_patterns=pats))
    # a pinned-revision download writes no refs/main, and KeySync loads these by NAME offline (HF_HUB_OFFLINE=1): from_pretrained("facebook/
    # hubert-base-ls960") then fails "couldn't connect" (11.11 VM smoke: face_gen_failed). The ref makes the name resolve to the pinned snapshot.
    ref = snap.parents[1] / "refs" / "main"; ref.parent.mkdir(parents=True, exist_ok=True); ref.write_text(rev)
print("keysync hub deps ok")
PY
dl https://www.adrianbulat.com/downloads/python-fan/s3fd-619a316812.pth $K/torch/hub/checkpoints/s3fd-619a316812.pth
dl https://www.adrianbulat.com/downloads/python-fan/2DFAN4-cd938726ad.zip $K/torch/hub/checkpoints/2DFAN4-cd938726ad.zip
printf "619a31681264d3f7f7fc7a16a42cbbe8b23f31a256f75a366e5a1bcd59b33543  $K/torch/hub/checkpoints/s3fd-619a316812.pth\ncd938726adb1f15f361263cce2db9cb820c42585fa8796ec72ce19107f369a46  $K/torch/hub/checkpoints/2DFAN4-cd938726ad.zip\n" | sha256sum -c -
if [ -z "${KS_KIT_DIR:-}" ] && [ -n "${KS_KIT_REPO:-}" ]; then   # r17 (Vast Serverless): our KeySync kit from a PRIVATE HF repo (HF_TOKEN), never in the public image
  KS_KIT_DIR=$M/cache/ks_kit; $PY -c "import sys; from huggingface_hub import snapshot_download; snapshot_download(sys.argv[1], local_dir=sys.argv[2], allow_patterns=['lora_pick.pt', 'lora_meta.json', 'kf.pt', 'interp.pt'])" "$KS_KIT_REPO" $KS_KIT_DIR     && echo "keysync kit from $KS_KIT_REPO" || echo "WARNING: the KeySync kit download from $KS_KIT_REPO failed"; fi
for f in student/lora_pick.pt student/lora_meta.json trt_cap/kf.pt trt_cap/interp.pt; do   # KS_KIT_DIR = a folder with these 4 files (HANDOFF / laptop chkpts)
  [ -s $K/$f ] || { [ -n "${KS_KIT_DIR:-}" ] && [ -s $KS_KIT_DIR/$(basename $f) ] && cp $KS_KIT_DIR/$(basename $f) $K/$f; } || echo "WARNING: no $K/$f (set KS_KIT_DIR): the KeySync face is unavailable"
done
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
