#!/bin/bash
# r16 TZ (5.1 / 10.2): ONE command -> a research box whose weights pass weights_check (KeySync = the r16 default, and LatentSync), clean set only.
#   = infra/docker/fetch_models.sh (the image's models volume: the same pinned list, models.lock.json) into $DUB_ROOT/models
#   + the links a research box runs with (no image env): KS_ROOT/code/pretrained_models/checkpoints, KS_ROOT/student, KS_ROOT/trt/cap,
#     the box's HF_HOME (Vast sets /workspace/.hf_home), TORCH_HOME hub (face_alignment), pyannote / pkuseg caches
#   + weights_check (keysync, auto) + models_lock check + the size of every set.
# Files already in the research places are ADOPTED (moved into models/, linked back): a set-up box downloads nothing again.
#     HF_TOKEN=<token, pyannote 3.1 + SVD terms accepted> (default: ~/.cache/huggingface/token; never printed)
#     KS_KIT_DIR=<folder with lora_pick.pt lora_meta.json kf.pt interp.pt> (the KeySync student + TRT capture: laptop chkpts / HANDOFF)
#     bash scripts/setup_weights.sh          exit 0 = weights_check passes for keysync and auto
# Never fetched: CodeFormer / ParseNet, htdemucs, InsightFace, RetinaFace (non-commercial); DUB_PROFILE is forced to clean here.
set -u
R=$(cd "$(dirname "$0")/.." && pwd); M=$R/models; K=$M/keysync; KS=${KS_ROOT:-/workspace/dub/keysync}; BOXHF=${HF_HOME:-}; PY=${DUB_ENV:-/venv/dabai}/bin/python
TH=${TORCH_HOME:-$HOME/.cache/torch}; t0=$(date +%s)
log(){ echo "[$(date +%T)] == $*"; }
[ -n "${HF_TOKEN:-}" ] || { [ -s $HOME/.cache/huggingface/token ] && export HF_TOKEN=$(cat $HOME/.cache/huggingface/token); }
[ -n "${HF_TOKEN:-}" ] || { echo "HF_TOKEN is needed (gated pyannote 3.1 + SVD VAE): set it or put it in ~/.cache/huggingface/token" >&2; exit 2; }
mkdir -p $K/checkpoints $K/student $K/trt_cap $K/torch/hub/checkpoints $M/cache
adopt(){   # adopt <research path> <models path>: a real file / dir that is only in the research place moves into models/ and is linked back AT ONCE
  if [ -e "$1" ] && [ ! -L "$1" ] && [ ! -e "$2" ]; then   # (same filesystem: mv = rename; a job starting meanwhile still finds the file)
    mkdir -p "$(dirname "$2")"; mv "$1" "$2" && ln -s "$2" "$1" && echo "adopted $1 -> $2"; fi; }
log "1/4 adopt what this box already has"
for f in WavLM-Base+.pt keyframe_dub.pt interpolation_dub.pt; do adopt $KS/code/pretrained_models/checkpoints/$f $K/checkpoints/$f; done
[ -d $KS/code/pretrained_models/.cache ] && [ ! -e $K/.cache ] && mv $KS/code/pretrained_models/.cache $K/.cache   # the hf local_dir metadata (no re-hash)
for f in lora_pick.pt lora_meta.json; do [ -e $KS/student/$f ] && [ ! -e $K/student/$f ] && cp -L $KS/student/$f $K/student/$f && echo "adopted (copy) $KS/student/$f"; done
for f in kf.pt interp.pt; do adopt $KS/trt/cap/$f $K/trt_cap/$f; done
for f in s3fd-619a316812.pth 2DFAN4-cd938726ad.zip; do adopt $TH/hub/checkpoints/$f $K/torch/hub/checkpoints/$f; done
adopt $TH/pyannote $M/cache/pyannote
if [ -n "$BOXHF" ] && [ "$BOXHF" != "$M/hf" ]; then
  for m in models--facebook--hubert-base-ls960 models--stabilityai--stable-video-diffusion-img2vid; do adopt $BOXHF/hub/$m $M/hf/hub/$m; done; fi
log "2/4 fetch the image's clean set into $M (infra/docker/fetch_models.sh; idempotent)"
( unset DUB_PROFILE; export DUB_ROOT=$R DUB_ENV=${DUB_ENV:-/venv/dabai}; bash $R/infra/docker/fetch_models.sh ) || log "fetch_models.sh reported a problem (see above); the checks below decide"
log "3/4 link the research-box places to models/"
lnk(){ mkdir -p "$(dirname "$2")"; if [ -d "$2" ] && [ ! -L "$2" ]; then rmdir "$2" 2>/dev/null || { echo "KEEP $2 (a non-empty dir, not replaced)"; return; }; fi; ln -sfn "$1" "$2"; }
[ -d $KS/code ] && lnk $K/checkpoints $KS/code/pretrained_models/checkpoints
lnk $K/student $KS/student; lnk $K/trt_cap $KS/trt/cap
for f in s3fd-619a316812.pth 2DFAN4-cd938726ad.zip; do mkdir -p $TH/hub/checkpoints; [ -e $TH/hub/checkpoints/$f ] || ln -s $K/torch/hub/checkpoints/$f $TH/hub/checkpoints/$f; done
[ -e $TH/pyannote ] || ln -s $M/cache/pyannote $TH/pyannote
[ -d $M/cache/pkuseg ] && [ ! -e $HOME/.pkuseg ] && ln -s $M/cache/pkuseg $HOME/.pkuseg
if [ -n "$BOXHF" ] && [ "$BOXHF" != "$M/hf" ]; then mkdir -p $BOXHF/hub
  for m in models--facebook--hubert-base-ls960 models--stabilityai--stable-video-diffusion-img2vid; do [ -e $BOXHF/hub/$m ] || ln -s $M/hf/hub/$m $BOXHF/hub/$m; done; fi
log "4/4 checks + sizes"
ok=1
for f in keysync auto; do KS_ROOT=$KS python3 $R/scripts/keysync/r16/weights_check.py --face $f --nc 0 > /tmp/wc_$f.txt 2>&1 && echo "weights_check --face $f: OK" || { ok=0; echo "weights_check --face $f: FAILED"; cat /tmp/wc_$f.txt; }; done
$PY $R/infra/docker/models_lock.py check --models $M || ok=0
echo "sizes (GB, links followed):"; for d in whisper-large-v3 qwen2.5-14b-instruct-gguf chatterbox latentsync sd-vae-ft-mse wav2vec2-base-960h mediapipe realesrgan spleeter cache/pyannote \
  keysync/checkpoints keysync/student keysync/trt_cap keysync/torch hf/hub/models--facebook--hubert-base-ls960 hf/hub/models--stabilityai--stable-video-diffusion-img2vid; do
  [ -e $M/$d ] && printf "  %-62s %6.2f\n" $d "$(du -sbL $M/$d 2>/dev/null | awk '{print $1 / 1e9}')"; done
printf "  %-62s %6.2f\n" TOTAL "$(du -sbL $M --exclude=trt_cache --exclude=inductor 2>/dev/null | awk '{print $1 / 1e9}')"
[ $ok = 1 ] && echo "SETUP_WEIGHTS_OK in $(( $(date +%s) - t0 )) s" || { echo "SETUP_WEIGHTS_FAILED (see above)"; exit 1; }
