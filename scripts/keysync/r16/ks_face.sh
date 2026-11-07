#!/bin/bash
# r16 T: the KeySync face stage of the e2e (DUB_FACE=keysync | auto -> keysync). The r16 chain accepted by Irina's eye (K1 r16-paste 5c07cf7,
# K2 r16-mouth 8d07fc6 / k2_k1chain_tag.sh, Q r16-qa 84031e1 route.json), per speaking face:
#   plan      ks_e2e_prep.py: route.json faces (lipsync with SILENCE outside own speech / source -> --others) or the dubbed-units fallback; no-face rule
#   crop      K2 tracked crop (k2_prep_track.py; KS_CROP=static = r15 ks_prep) + K1 pad: the last KS_PAD (60) frames mirrored, the seam frame not repeated
#   ref       K2 k2_refpick.py -> the reference frame (KS_REF=auto | <frame> | 0)
#   [GPU]     delay probe (KS_DELAY=probe | <ms>): KeySync on KS_PROBE_S (10) s from the face's first own speech, SyncNet AV offset of the raw render
#             vs the dub it heard -> delay = -offset x 40 ms (old03 0, m03 / m18 40); the full KeySync (fast2 + K2 gate KS_GATE) on the padded crop with
#             the delayed + padded dub; check: SyncNet of the full raw render -> if still != 0, the paste shift = -offset (K1 fallback)
#   post      ECC align (k_ecc) -> ks_tsmooth (KS_TSMOOTH gate:0.5) -> ks_snap -> ks_guard -> K2 pause lip-close warp (KS_PAUSE=warp | off)
#   paste     ks_plain (K1 + K2): ECC align, scale locked, lfm 6, upper-src, keep-nose, tex 2, others-grow 1.3, --absent skip; KS_PASTE_JOBS workers
# usage: ks_face.sh <id> <src video> <audio dir: dubbed_16k.wav + final_mix_aac.m4a> <work dir> <out.mp4> [route.json]
# exit 0 = OUT written; 3 = nothing to lip-sync (keep the source picture, not an error); other = failed.
# Env: KS_DELAY KS_PROBE_S KS_REF KS_CROP KS_PAD KS_GATE KS_TSMOOTH KS_PAUSE KS_PASTE_SHIFT KS_PLAIN_ARGS KS_PLAIN_EXTRA KS_INTERP_STEPS KS_SYNC_MAX_S
#      KS_PASTE_JOBS KS_HELD_FD KS_MIN_IOD KS_NUM_FACES
set -o pipefail
ID=$1; SRC=$2; AD=$3; W=$4; OUTV=$5; ROUTE=${6:-}
R=$(cd "$(dirname "$0")/../../.." && pwd); KS=${KS_ROOT:-/workspace/dub/keysync}; LOCK=${KS_GPU_LOCK:-/workspace/dub/gpu.lock}
PY=${DUB_PY:-/venv/dabai/bin/python}; MPY=${DUB_MEDIAPIPE_PY:-/venv/mediapipe/bin/python}; LORA=${KS_LORA:-$KS/student/lora_pick.pt}
KW=$W/ks; rm -rf $KW; mkdir -p $KW/data $KW/out $KW/probe $KW/kit
for f in $R/scripts/keysync/*.py $R/scripts/keysync/r15/*.py $R/scripts/keysync/r16/*.py; do cp $f $KW/kit/; done   # one flat kit dir = this checkout's code (copies: python resolves a symlinked script's imports to its real dir)
KT=$KW/kit; PAD=${KS_PAD:-60}
t() { echo "KS_T $1 $(date +%s.%N)"; }
dur() { ffprobe -v error -show_entries format=duration -of csv=p=0 "$1"; }
sync_off() {   # sync_off <video> <wav> <json>: SyncNet global AV offset of a raw KeySync render (first KS_SYNC_MAX_S s); prints the offset or "x"
  local V=$1 A=$2 J=$3 MX=${KS_SYNC_MAX_S:-60}
  if [ "$MX" != 0 ] && python3 -c "import sys; sys.exit(0 if float(sys.argv[1]) > float(sys.argv[2]) else 1)" "$(dur $V)" "$MX"; then
    ffmpeg -v error -y -i $V -t $MX -an -c:v copy ${J%.json}.mp4 && ffmpeg -v error -y -i $A -t $MX ${J%.json}.wav && V=${J%.json}.mp4 A=${J%.json}.wav; fi
  ( cd $R && $PY scripts/quality/sync_perframe.py --video $V --audio $A --out $J ) > ${J%.json}.log 2>&1 \
    && python3 -c "import json; print(int(json.load(open('$J'))['global']['av_offset']))" || echo x; }

t master; ffmpeg -v error -y -i "$SRC" -map 0:v:0 -an -vf fps=25 -c:v libx264 -preset fast -crf 10 -pix_fmt yuv420p $KW/master_25fps.mp4 || exit 1
N=$(ffprobe -v error -count_packets -select_streams v:0 -show_entries stream=nb_read_packets -of csv=p=0 $KW/master_25fps.mp4)
t landmarks   # CPU; up to KS_NUM_FACES faces (route.json faces are matched to these tracks)
GLOG_minloglevel=2 OMP_NUM_THREADS=4 $MPY $R/scripts/pilot/mp_landmarks.py $KW/master_25fps.mp4 $KW/master_mp.npz --num-faces ${KS_NUM_FACES:-4} > $KW/mp.log 2>&1 || { tail -3 $KW/mp.log; exit 1; }
t plan; $PY $R/scripts/keysync/r16/ks_e2e_prep.py --mp $KW/master_mp.npz --dub $AD/dubbed_16k.wav --out $KW ${ROUTE:+--route $ROUTE} | tee $KW/plan.log | cut -c1-600 || exit 1
TAGS=$(python3 -c "import json; print(' '.join(f['tag'] for f in json.load(open('$KW/faces.json'))['faces']))")
[ -n "$TAGS" ] || { echo "KS_FACE_KEEP_SOURCE $ID: no face to lip-sync ($(python3 -c "import json; d=json.load(open('$KW/faces.json')); print(d['mode'], d.get('skipped'))"))"; exit 3; }

t crop
for f in $TAGS; do
  if [ "${KS_CROP:-track}" = track ]; then   # K2: constant side 1.8 face heights, the centre follows the face (smoothed)
    TK=${KS_TRACK_K:-1.8}; [ -s $KW/others_$f.npz ] && TK=${KS_TRACK_K2:-1.25}   # another face in the shot: the tight crop (r15 K m03: 1.25), so KeySync / SyncNet see one face
    $MPY $KT/k2_prep_track.py --master $KW/master_25fps.mp4 --lm $KW/lm_$f.npz --out $KW/data/crop_$f --k $TK > $KW/crop_$f.log 2>&1 || { tail -3 $KW/crop_$f.log; exit 1; }
    mv $KW/data/crop_${f}_box.json $KW/data/box_$f.json
  else
    ( mkdir -p $KW/data/st/videos $KW/data/st/audios; cd $KW && $PY $KT/ks_prep.py $KW/master_25fps.mp4 $KW/lm_$f.npz $KW/dub_$f.wav $KW/data/st $f ) > $KW/crop_$f.log 2>&1 || { tail -3 $KW/crop_$f.log; exit 1; }
    mv $KW/data/st/videos/$f.mp4 $KW/data/crop_$f.mp4; mv $KW/data/st/box_$f.json $KW/data/box_$f.json
  fi
  $MPY - $KW/data/crop_$f.mp4 $KW/data/pad_$f.mp4 $PAD <<'PY' >> $KW/crop_$f.log 2>&1 || exit 1
import subprocess, sys, cv2   # K1 5c07cf7: the last PAD frames mirrored WITHOUT repeating the seam frame (KeySync degenerated on a duplicate)
src, out, pad = sys.argv[1], sys.argv[2], int(sys.argv[3]); c = cv2.VideoCapture(src); F = []
while True:
    ok, f = c.read()
    if not ok: break
    F.append(f)
F = F + F[-2:-2 - pad:-1]
p = subprocess.Popen(["ffmpeg", "-v", "error", "-y", "-f", "rawvideo", "-pix_fmt", "bgr24", "-s", "512x512", "-r", "25", "-i", "-", "-c:v", "libx264", "-crf", "10",
                      "-pix_fmt", "yuv420p", out], stdin=subprocess.PIPE)
for f in F: p.stdin.write(f.tobytes())
p.stdin.close(); p.wait(); print("pad", len(F), "frames")
PY
  $MPY $R/scripts/quality/mp_landmarks.py --video $KW/data/crop_$f.mp4 --out $KW/crop_lm_$f.npz > /dev/null 2>&1 || { echo "KS: crop landmarks failed ($f)"; exit 1; }
  REF=${KS_REF:-auto}
  if [ "$REF" = auto ]; then   # K2: a frame with the upper teeth showing as KeySync's identity (black-hole mouths roughly halved); 0 if the pick fails
    REF=$($MPY $KT/k2_refpick.py --video $KW/data/crop_$f.mp4 --lm $KW/crop_lm_$f.npz --out $KW/ref_$f 2> $KW/ref_$f.log | grep "^K2_REFPICK" | python3 -c "import json,sys; print(json.loads(sys.stdin.read()[11:])['pick'])" 2>/dev/null)
    [ -n "$REF" ] || { echo "KS WARN refpick failed ($f): ref 0"; REF=0; }
  fi
  echo $REF > $KW/ref_$f.txt
done

# ---- GPU section = ONE hold of the GPU lock (engines, probe, full KeySync, SyncNet checks); released before the CPU post / paste.
# KS_HELD_FD: the caller (e2e.sh with R10_HEAVY_LOCK) already holds the lock on that fd since its audio stage: no second wait in the queue.
if [ -n "${KS_HELD_FD:-}" ]; then REL="flock -u $KS_HELD_FD"; else t wait_gpu; exec 7>$LOCK; flock 7; REL="flock -u 7"; fi
export KS_GPU_LOCK=/dev/null   # (ks_engines.sh: the lock is held here)
t engines; ENG=$(bash $R/scripts/keysync/r16/ks_engines.sh 2> $KW/engines.log | tail -1); [ -d "$ENG" ] || { tail -5 $KW/engines.log; exit 1; }
LMK=${KS_LMK_CACHE:-$KS/lmk}; mkdir -p $LMK; [ -n "${KS_TORCH_HOME:-}" ] && export TORCH_HOME=$KS_TORCH_HOME   # (image: face_alignment's weights on the volume)
ksfast() {   # ksfast <jobs> <out> <log> [gate]: the resident KeySync-fast worker (fast2: TRT keyframes + 2-step LCM student, compiled VAE decode)
  ( cd $KS/code && . ../venv/bin/activate && env KS_DC=2:0 KS_LMK_CACHE=$LMK TORCHINDUCTOR_CACHE_DIR=${TORCHINDUCTOR_CACHE_DIR:-$KS/inductor} ${4:+KS_GATE=$4} \
    python $KT/ks_fast.py --jobs $1 --out $2 --cfg _s8 --vae-half --compile unet --trt $ENG --interp-lora $LORA --interp-steps ${KS_INTERP_STEPS:-2} \
      --vae-batch 8 --vae-compile ) > $3 2>&1 || { tail -5 $3; return 1; }; grep "^KS_JOB" $3 | cut -c1-400; }
declare -A DLY
if [ "${KS_DELAY:-probe}" = probe ]; then
  t probe; : > $KW/probe/jobs.txt; PS=${KS_PROBE_S:-10}
  for f in $TAGS; do   # the window: KS_PROBE_S s from the first own speech of this face (the dub it hears, silence elsewhere)
    S0=$(python3 - $KW/dub_$f.wav $PS <<'PY'
import sys, wave, numpy as np
w = wave.open(sys.argv[1]); sr = w.getframerate(); x = np.frombuffer(w.readframes(w.getnframes()), np.int16).astype(np.float32); ps = float(sys.argv[2])
e = np.sqrt(np.convolve(x ** 2, np.ones(sr // 10) / (sr // 10), "same")); on = np.flatnonzero(e > 0.05 * (e.max() + 1e-6))
print(max(0.0, min((on[0] / sr - 0.5) if len(on) else 0.0, len(x) / sr - ps)))
PY
)
    ffmpeg -v error -y -ss $S0 -i $KW/data/crop_$f.mp4 -t $PS -c:v libx264 -crf 10 -pix_fmt yuv420p $KW/probe/v_$f.mp4 && ffmpeg -v error -y -ss $S0 -i $KW/dub_$f.wav -t $PS $KW/probe/a_$f.wav || exit 1
    echo "$f $KW/probe/v_$f.mp4 $KW/probe/a_$f.wav" >> $KW/probe/jobs.txt
  done
  ksfast $KW/probe/jobs.txt $KW/probe/out $KW/probe/ks_fast.log || exit 1
  for f in $TAGS; do O=$(sync_off $(ls $KW/probe/out/$f/*.mp4 | head -1) $KW/probe/a_$f.wav $KW/probe/sync_$f.json)
    [ "$O" = x ] && { echo "KS WARN probe sync failed ($f): delay 0"; O=0; }; DLY[$f]=$(( -O * 40 )); echo "KS_PROBE $f av_offset $O -> dub delay ${DLY[$f]} ms"; done
else for f in $TAGS; do DLY[$f]=${KS_DELAY}; done; fi
t keysync; : > $KW/jobs.txt
for f in $TAGS; do   # the dub KeySync hears: delayed (or advanced) by the probe, padded with silence to the padded crop (the final mix is never touched)
  D=${DLY[$f]}; PD=$(python3 -c "print(($N + $PAD) / 25)")
  if [ "$D" -ge 0 ]; then AF="adelay=$D:all=1,apad=whole_dur=$PD"; else AF="atrim=start=$(python3 -c "print(-$D / 1000)"),asetpts=PTS-STARTPTS,apad=whole_dur=$PD"; fi
  ffmpeg -v error -y -i $KW/dub_$f.wav -af "$AF" -t $PD -ar 16000 -ac 1 $KW/data/dubp_$f.wav || exit 1
  ffmpeg -v error -y -i $KW/dub_$f.wav -af "apad=whole_dur=$PD" -t $PD -ar 16000 -ac 1 $KW/data/dubo_$f.wav || exit 1   # the ORIGINAL timing, for the check
  if [ "$N" -gt "$(( ${KS_CHUNK_MIN_S:-90} * 25 ))" ]; then   # long video: ~KS_CHUNK_S pieces cut at dub pauses, the same reference in each (RAM flat)
    $PY $KT/ks_chunk.py split --video $KW/data/pad_$f.mp4 --audio $KW/data/dubp_$f.wav --ref $(cat $KW/ref_$f.txt) --chunk-s ${KS_CHUNK_S:-60} --out $KW/data/chunks_$f --tag $f \
      >> $KW/jobs.txt 2> $KW/chunk_$f.log || { tail -3 $KW/chunk_$f.log; exit 1; }; grep KS_CHUNKS $KW/chunk_$f.log
  else echo "$f $KW/data/pad_$f.mp4 $KW/data/dubp_$f.wav $(cat $KW/ref_$f.txt)" >> $KW/jobs.txt; fi
done
ksfast $KW/jobs.txt $KW/out $KW/ks_fast.log "${KS_GATE-thr=0.12,cap=0.25,hi=8,lo=2}" || exit 1
for f in $TAGS; do [ -d $KW/data/chunks_$f ] || continue; mkdir -p $KW/out/$f
  $PY $KT/ks_chunk.py join --dir $KW/data/chunks_$f --renders $KW/out --out $KW/out/$f/joined.mp4 --tag $f || exit 1; done
t check; declare -A SHF
for f in $TAGS; do SH=${KS_PASTE_SHIFT:-auto}
  if [ "$SH" = auto ]; then O=$(sync_off $(ls $KW/out/$f/*.mp4 | head -1) $KW/data/dubo_$f.wav $KW/sync_$f.json)   # vs the ORIGINAL dub (K2): still off after the delay -> paste shift (K1 fallback)
    [ "$O" = x ] && O=0; SH=$(( -O )); fi
  SHF[$f]=$SH; echo "KS_CHECK $f delay ${DLY[$f]} ms, paste shift $SH"
done
$REL   # GPU done: the post + paste below are CPU only

CUR=$KW/master_25fps.mp4; n=0
for f in $TAGS; do
  RAW=$(ls $KW/out/$f/*.mp4 | head -1); P=$KW/post_$f; mkdir -p $P
  t post_$f
  $MPY $R/scripts/quality/mp_landmarks.py --video $RAW --out $P/raw_lm.npz > $P/raw_lm.log 2>&1 || { tail -3 $P/raw_lm.log; exit 1; }
  $PY $KT/$([ "${KS_ECC_PAR:-1}" = 1 ] && echo "k_ecc_par.py --warm 30" || echo k_ecc.py) --ks $RAW --src $KW/data/pad_$f.mp4 --lm $KW/crop_lm_$f.npz --out $P/ecc.npz > $P/ecc.log 2>&1 || { tail -3 $P/ecc.log; exit 1; }
  L=$RAW
  if [ "${KS_TSMOOTH:-gate:0.5}" != off ]; then $PY $KT/ks_tsmooth.py --layer $L --lm $P/raw_lm.npz --src-lm $KW/crop_lm_$f.npz --out $P/ts.mp4 --alpha ${KS_TSMOOTH:-gate:0.5} > $P/ts.log 2>&1 \
    && L=$P/ts.mp4 || { tail -3 $P/ts.log; exit 1; }; grep -h "gate:" $P/ts.log | head -1; fi
  $PY $KT/ks_snap.py --layer $L --lm $P/raw_lm.npz --out $P/snap.mp4 > $P/snap.log 2>&1 && L=$P/snap.mp4 || { tail -3 $P/snap.log; exit 1; }
  $PY $KT/ks_guard.py --layer $L --src $KW/data/pad_$f.mp4 --n $N --out $P/guard.mp4 > $P/guard.log 2>&1 && L=$P/guard.mp4 || { tail -3 $P/guard.log; exit 1; }
  if [ "${KS_PAUSE:-warp}" = warp ]; then   # K2 pause (b): the lips closed in the dub's pauses (the layer is on the dub timeline: shift 0)
    $MPY $KT/k2_pause.py --mode warp --ks $L --ks-lm $P/raw_lm.npz --wav $KW/dub_$f.wav --shift 0 --out $P/pause.mp4 > $P/pause.log 2>&1 && L=$P/pause.mp4 || { tail -3 $P/pause.log; exit 1; }; fi
  t paste_$f; n=$((n + 1)); NXT=$KW/paste_$n.mp4
  OTH=""; [ -s $KW/others_$f.npz ] && OTH="--others $KW/others_$f.npz --others-grow ${KS_OTHERS_GROW:-1.3}"
  ARGS="--ks $L --ks-lm $P/raw_lm.npz --master $CUR --box $KW/data/box_$f.json --lm $KW/lm_$f.npz --audio $AD/final_mix_aac.m4a --shift ${SHF[$f]} --absent skip $OTH \
        ${KS_PLAIN_ARGS---lfm 6 --upper-src 0.12:0.32 --keep-nose 0.04 --align 2 --align-ecc $P/ecc.npz --align-scale 0 --tex 2} ${KS_PLAIN_EXTRA:-}"
  if [ "${KS_PASTE_JOBS:-0}" != 1 ] && [ -f $KT/ks_plain_par.py ]; then
    ( cd $KT && KS_CMS=1 $PY $KT/ks_plain_par.py --jobs ${KS_PASTE_JOBS:-0} --out $NXT -- $ARGS ) > $P/plain.log 2>&1 || { tail -5 $P/plain.log; exit 1; }
  else ( cd $KT && KS_CMS=1 $PY $KT/ks_plain.py $ARGS --out $NXT ) > $P/plain.log 2>&1 || { tail -5 $P/plain.log; exit 1; }; fi
  echo "KS_PASTE $f delay ${DLY[$f]} ms shift ${SHF[$f]} ref $(cat $KW/ref_$f.txt) $(grep -E 'plain paste:' $P/plain.log | head -1)"; CUR=$NXT
done
t end; mv $CUR $OUTV && echo "KS_FACE_DONE $ID faces $TAGS"
