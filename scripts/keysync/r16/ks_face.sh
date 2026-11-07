#!/bin/bash
# r16 T: the KeySync-fast face stage of the e2e (DUB_FACE=keysync; checklist #25):
#   25 fps master -> MediaPipe (2 faces) -> speaker -> face plan (#14, ks_e2e_prep.py) -> ks_prep crop per speaking face
#   -> ks_fast (fast2: TRT keyframes + TRT 2-step LCM student, compiled VAE decode; engines built on first use, ks_engines.sh)
#   -> paste shift (SyncNet AV offset of the raw output, or a fixed KS_PASTE_SHIFT) -> ks_plain paste, one face after the other.
# usage: ks_face.sh <id> <src video> <audio dir with dubbed_16k.wav + final_mix_aac.m4a> <work dir> <out.mp4>   (exit != 0 = face stage failed)
# Hooks (env): KS_DUB_DELAY_MS (0)   the dub fed to KeySync delayed by N ms (K1's 40 ms fix; the final mix is untouched)
#              KS_PASTE_SHIFT (auto) auto = -SyncNet av_offset of the raw KeySync output; an integer = that shift (K1: 0)
#              KS_PLAIN_ARGS ("--tex 2 --cavity 0.8")  extra ks_plain.py args (K1 / K2 drop their paste flags here)
#              KS_INTERP_STEPS (2)   KS_BOX_K / KS_BOX_M (2 co-occurring faces: 1.25 / 0.4 as r15 K m03)
set -o pipefail
ID=$1; SRC=$2; AD=$3; W=$4; OUTV=$5
R=$(cd "$(dirname "$0")/../../.." && pwd); KS=${KS_ROOT:-/workspace/dub/keysync}; LOCK=${KS_GPU_LOCK:-/workspace/dub/gpu.lock}
PY=${DUB_PY:-/venv/dabai/bin/python}; MPY=${DUB_MEDIAPIPE_PY:-/venv/mediapipe/bin/python}; LORA=${KS_LORA:-$KS/student/lora_pick.pt}
KW=$W/ks; rm -rf $KW; mkdir -p $KW/data/videos $KW/data/audios $KW/out
t() { echo "KS_T $1 $(date +%s.%N)"; }
t master; ffmpeg -v error -y -i "$SRC" -map 0:v:0 -an -vf fps=25 -c:v libx264 -preset fast -crf 10 -pix_fmt yuv420p $KW/master_25fps.mp4 || exit 1
t landmarks   # CPU; the 2-face pass also gives the largest face per frame ('lm')
GLOG_minloglevel=2 OMP_NUM_THREADS=4 $MPY $R/scripts/pilot/mp_landmarks.py $KW/master_25fps.mp4 $KW/master_mp2.npz --num-faces 2 > $KW/mp.log 2>&1 || { tail -3 $KW/mp.log; exit 1; }
t plan; $PY $R/scripts/keysync/r16/ks_e2e_prep.py --mp $KW/master_mp2.npz --dub $AD/dubbed_16k.wav --out $KW | tee $KW/plan.log | tail -1 || exit 1
TAGS=$(python3 -c "import json; print(' '.join(f['tag'] for f in json.load(open('$KW/faces.json'))['faces']))")
MODE=$(python3 -c "import json; print(json.load(open('$KW/faces.json'))['mode'])")
[ -n "$TAGS" ] || { echo "KS: no speaking face"; exit 1; }
t crop; : > $KW/jobs.txt
for f in $TAGS; do
  if [ "$MODE" = multi_face ]; then export KS_BOX_K=${KS_BOX_K:-1.25} KS_BOX_M=${KS_BOX_M:-0.4}; fi
  ( cd $KW && $PY $KS/ks_prep.py $KW/master_25fps.mp4 $KW/lm_$f.npz $KW/dub_$f.wav $KW/data $f ) > $KW/prep_$f.log 2>&1 || { tail -3 $KW/prep_$f.log; exit 1; }
  echo "$f $KW/data/videos/$f.mp4 $KW/data/audios/$f.wav" >> $KW/jobs.txt
done
# GPU section = ONE hold of the GPU lock (TensorRT engines on first use + KeySync + the SyncNet offsets), released before the CPU paste.
# KS_HELD_FD: the caller (e2e.sh with R10_HEAVY_LOCK) already holds the lock on that fd since its audio stage: no second wait in the queue.
if [ -n "${KS_HELD_FD:-}" ]; then REL="flock -u $KS_HELD_FD"; else t wait_gpu; exec 7>$LOCK; flock 7; REL="flock -u 7"; fi
export KS_GPU_LOCK=/dev/null   # (ks_engines.sh: the lock is held here)
t engines; ENG=$(bash $R/scripts/keysync/r16/ks_engines.sh 2> $KW/engines.log | tail -1); [ -d "$ENG" ] || { tail -5 $KW/engines.log; exit 1; }
t keysync   # the resident worker: one process for every speaking face of this video
LMK=${KS_LMK_CACHE:-$KS/lmk}; mkdir -p $LMK; [ -n "${KS_TORCH_HOME:-}" ] && export TORCH_HOME=$KS_TORCH_HOME   # (image: face_alignment's weights on the volume)
( cd $KS/code && . ../venv/bin/activate && KS_DC=2:0 KS_LMK_CACHE=$LMK TORCHINDUCTOR_CACHE_DIR=${TORCHINDUCTOR_CACHE_DIR:-$KS/inductor} \
  python ../ks_fast.py --jobs $KW/jobs.txt --out $KW/out --cfg _s8 --vae-half --compile unet --trt $ENG \
    --interp-lora $LORA --interp-steps ${KS_INTERP_STEPS:-2} --vae-batch 8 --vae-compile ) > $KW/ks_fast.log 2>&1 || { tail -5 $KW/ks_fast.log; exit 1; }
grep "^KS_JOB" $KW/ks_fast.log | tail -3
t shift; declare -A SHF
for f in $TAGS; do
  KO=$KW/out/$f/${f}_$f.mp4; [ -s $KO ] || { echo "KS: no output for $f"; exit 1; }; SH=${KS_PASTE_SHIFT:-auto}
  if [ "$SH" = auto ]; then   # = r15 K: the raw output's SyncNet AV offset (vs the audio KeySync heard), pasted with shift = -offset
    SV=$KO; SA=$KW/data/audios/$f.wav; MX=${KS_SYNC_MAX_S:-60}   # one global offset: the first MX s are enough (5 min: 438 -> ~90 GPU-s)
    if [ "$MX" != 0 ] && python3 -c "import sys; sys.exit(0 if float(sys.argv[1]) > float(sys.argv[2]) else 1)" "$(ffprobe -v error -show_entries format=duration -of csv=p=0 $KO)" "$MX"; then
      SV=$KW/sync_$f.mp4; SA=$KW/sync_$f.wav
      ffmpeg -v error -y -i $KO -t $MX -an -c:v copy $SV && ffmpeg -v error -y -i $KW/data/audios/$f.wav -t $MX $SA || { SV=$KO; SA=$KW/data/audios/$f.wav; }
    fi
    ( cd $R && $PY scripts/quality/sync_perframe.py --video $SV --audio $SA --out $KW/raw_sync_$f.json ) > $KW/raw_sync_$f.log 2>&1 \
      && SH=$(python3 -c "import json; print(-int(json.load(open('$KW/raw_sync_$f.json'))['global']['av_offset']))") || { echo "KS WARN sync offset failed ($f): shift 0"; SH=0; }
  fi
  SHF[$f]=$SH
done
$REL   # GPU done: the paste below is CPU only
CUR=$KW/master_25fps.mp4; n=0
for f in $TAGS; do
  KO=$KW/out/$f/${f}_$f.mp4; SH=${SHF[$f]}
  t paste_$f; $MPY $R/scripts/quality/mp_landmarks.py --video $KO --out $KW/ks_lm_$f.npz > $KW/ks_lm_$f.log 2>&1 || { tail -3 $KW/ks_lm_$f.log; exit 1; }
  OTH=""; [ -s $KW/others_$f.npz ] && OTH="--others $KW/others_$f.npz"; n=$((n + 1)); NXT=$KW/paste_$n.mp4
  ( cd $KS && KS_CMS=1 $PY ks_plain.py --ks $KO --master $CUR --box $KW/data/box_$f.json --lm $KW/lm_$f.npz $OTH --shift $SH --audio $AD/final_mix_aac.m4a \
      --out $NXT --ks-lm $KW/ks_lm_$f.npz ${KS_PLAIN_ARGS---tex 2 --cavity 0.8} ) > $KW/plain_$f.log 2>&1 || { tail -5 $KW/plain_$f.log; exit 1; }
  echo "KS_PASTE $f shift $SH $(grep -E 'plain paste:' $KW/plain_$f.log | head -1)"; CUR=$NXT
done
t end; mv $CUR $OUTV && echo "KS_FACE_DONE $ID mode $MODE faces $TAGS"
