#!/bin/bash
# r16 Q: the CURRENT face path on one clip after its e2e audio: KeySync fast2 (student 2 steps + TRT, RECIPE r15 K) + the plain paste (ks_plain.py),
# then the r16 detectors + the 0.5x flag clips.  Needs T's engines (/workspace/dub/keysync/READY).
#   bash scripts/r16q/q_face.sh prep <id> <clip.mp4>      CPU: master 25 fps, landmarks, static crop, dub +40 ms (STEP 0), other faces
#   bash scripts/r16q/q_face.sh gen <id> [<id> ...]       GPU (gpu.lock): ONE resident ks_fast worker for all ids
#   bash scripts/r16q/q_face.sh post <id>                 sync offset (GPU lock, short) -> ks_plain -> mux the accepted mix -> detectors -> flag clips
set -u; MODE=$1; shift
K=/workspace/dub/keysync; Q=/workspace/dub/r16q; A=/workspace/dub/artifacts/r4a; R=/workspace/dub/dubai; LORA=/workspace/dub/keysync/student/lora_pick.pt; ENG=/workspace/dub/keysync/trt/cache/NVIDIA_GeForce_RTX_5090_575.57.08_trt10.16.1.11/eng_s
PYD=/venv/dabai/bin/python; PYM=/venv/mediapipe/bin/python
case $MODE in
prep)
  id=$1; V=$2; W=$Q/ks/$id; mkdir -p $W
  ffmpeg -nostdin -v error -y -i $V -vf fps=25 -c:v libx264 -crf 12 -pix_fmt yuv420p -c:a aac -b:a 192k $W/master_25fps.mp4
  $PYM $R/scripts/quality/mp_landmarks.py --video $W/master_25fps.mp4 --out $W/src_mp.npz | tail -1
  $PYM $K/k_faces2.py --video $W/master_25fps.mp4 --speaker-lm $W/src_mp.npz --out $W/faces2.npz > $W/faces2.log 2>&1 || echo "WARN faces2 $id"
  bash $K/../dubai/scripts/keysync/ks_prep_src.sh $id $W/master_25fps.mp4 2>&1 | tail -1
  D=$A/r9_$id/$id/dubbed_16k.wav; [[ $id == *_src ]] && D=$K/lm/${id}_src16k.wav   # <id>_src = PROXY: KeySync on the source (Russian) speech, no dub
  [ -f $D ] || { echo "NO DUB $id"; exit 1; }
  cp $D $W/dub16k_base.wav; bash $0 redub $id 0
  echo "PREP_DONE $id $(cat $K/data/box_$id.json)";;
redub)   # the generator input = the base dub delayed by <ms> (negative = advanced); the final mix is untouched
  id=$1; ms=$2; W=$Q/ks/$id
  if [ "$ms" -gt 0 ]; then AF="adelay=$ms:all=1"; elif [ "$ms" -lt 0 ]; then AF="atrim=start=$(python3 -c "print(-$ms/1000)"),asetpts=PTS-STARTPTS"; else AF=anull; fi
  ffmpeg -nostdin -v error -y -i $W/dub16k_base.wav -af $AF -ar 16000 -ac 1 $K/data/audios/$id.wav; echo $ms > $W/delay_ms; rm -f $W/raw_sync.json $W/ks_mp.npz;;
gen)
  J=$Q/ks/jobs_$$.txt; : > $J; for id in "$@"; do echo "$id $K/data/videos/$id.mp4 $K/data/audios/$id.wav" >> $J; done
  cd $K/code; . ../venv/bin/activate
  TORCHINDUCTOR_CACHE_DIR=$K/inductor KS_DC=2:0 KS_LMK_CACHE=$K/lmk flock /workspace/dub/gpu.lock python ../ks_fast.py --jobs $J --out $Q/ks/out --cfg _s8 --vae-half --compile unet --trt $ENG \
    --interp-lora $LORA --interp-steps 2 --vae-batch 8 --vae-compile 2>&1 | grep -E "KS_JOB|Error|error" | cut -c1-400; echo "GEN_DONE $*";;
post)
  id=$1; W=$Q/ks/$id; KS=$Q/ks/out/$id/${id}_$id.mp4; [ -f $KS ] || { echo "NO KS $id"; exit 1; }
  cd $R; . scripts/env.sh >/dev/null 2>&1; export DUB_ROOT=$R DUB_ALLOW_NC=1
  [ -s $W/raw_sync.json ] || flock /workspace/dub/gpu.lock python scripts/quality/sync_perframe.py --video $KS --audio $K/data/audios/$id.wav --out $W/raw_sync.json > $W/raw_sync.log 2>&1
  MEAS=$(python3 -c "import json; print(-int(json.load(open(\"$W/raw_sync.json\"))[\"global\"][\"av_offset\"]))"); SH=0   # coordinator 07.11: measure the raw KeySync AV offset; if != 0, redub by it and regenerate, then paste --shift 0
  echo "$MEAS" > $W/meas_$(cat $W/delay_ms).txt
  if [ -z "${Q_NOREGEN:-}" ] && [ "$MEAS" != 0 ] && [ "$(cat $W/delay_ms)" = 0 ]; then echo "REGEN $id $((MEAS * 40))"; exit 2; fi
  [ -s $W/ks_mp.npz ] || $PYM scripts/quality/mp_landmarks.py --video $KS --out $W/ks_mp.npz > /dev/null
  MIX=$A/r9_${id}_htd/$id/final_mix_aac.m4a; [ -f $MIX ] || MIX=$A/r9_$id/$id/final_mix_aac.m4a; DUB=$A/r9_$id/$id/dubbed_16k.wav
  [[ $id == *_src ]] && { MIX=$K/lm/${id}_src.m4a; DUB=$K/lm/${id}_src16k.wav; }
  OTH=""; python3 -c "import numpy as np,sys; o=np.load(\"$W/faces2.npz\")[\"others\"]; sys.exit(0 if np.isfinite(o).any(axis=(2,3)).any(axis=1).mean() > 0.05 else 1)" 2>/dev/null && OTH="--others $W/faces2.npz"
  cd $K; KS_CMS=1 $PYD ks_plain.py --ks $KS --master $W/master_25fps.mp4 --box data/box_$id.json --lm $W/src_mp.npz $OTH --shift $SH --audio $MIX --out $W/final.mp4 \
    --ks-lm $W/ks_mp.npz > $W/plain.log 2>&1 || { echo "FAIL plain $id"; tail -3 $W/plain.log; exit 1; }
  O=/workspace/dub/outbox/r16_q/$id; mkdir -p $O; cp $W/final.mp4 $O/${id}_keysync_fast2plain.mp4
  cd $R; flock /workspace/dub/gpu.lock python scripts/quality/sync_perframe.py --video $W/final.mp4 --audio $DUB --out $W/final_sync.json > $W/final_sync.log 2>&1
  $PYM scripts/quality/r16q_detect.py --src $W/master_25fps.mp4 --ours $W/final.mp4 --out $W/det.json --npz $W/det.npz --label $id | cut -c1-600
  python3 scripts/quality/r16q_clips.py --det $W/det.json --npz $W/det.npz --id $id --out $O
  echo "POST_DONE $id delay_ms $(cat $W/delay_ms) measured_after $MEAS others ${OTH:+yes}";;
esac
