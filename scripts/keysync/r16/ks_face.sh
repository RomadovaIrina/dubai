#!/bin/bash
# r16 T: the KeySync face stage of the e2e (DUB_FACE=keysync | auto -> keysync). The r16 chain accepted by Irina's eye (K1 r16-paste 5c07cf7,
# K2 r16-mouth 8d07fc6 / k2_k1chain_tag.sh, Q r16-qa 84031e1 route.json), per speaking face:
#   plan      ks_e2e_prep.py: route.json faces (lipsync with SILENCE outside own speech / source -> --others) or the dubbed-units fallback; no-face rule
#   crop      KS_CROP=auto: K2 tracked crop (k2_prep_track.py) if the static face share < 0.45 or the head travels > 0.4 face heights, else the r15 static crop + K1 pad: the last KS_PAD (60) frames mirrored, the seam frame not repeated
#   ref       KS_REF=0 (default, Irina 08.11: frame 0 of this run = of the shot in multi-shot) | auto (K2 k2_refpick.py) | <frame>
#   [GPU]     delay probe (KS_DELAY=probe | <ms>): KeySync on KS_PROBE_S (10) s from the face's first own speech, SyncNet AV offset of the raw render
#             vs the dub it heard -> delay = -offset x 40 ms (old03 0, m03 / m18 40); the full KeySync (fast2 + K2 gate KS_GATE) on the padded crop with
#             the delayed + padded dub; check: SyncNet of the full raw render -> if still != 0, the paste shift = -offset (K1 fallback)
#   post      ECC align (k_ecc) -> ks_tsmooth (KS_TSMOOTH gate:0.5) -> ks_snap -> ks_guard -> ks_seam (KS_SEAM on) -> K2 pause lip-close warp (KS_PAUSE=warp; default off: Irina pending)
#   paste     ks_plain (K1 + K2): ECC align, scale locked, lfm 6, upper-src, keep-nose, tex 2, others-grow 1.3, --absent skip; KS_PASTE_JOBS workers
# usage: ks_face.sh <id> <src video> <audio dir: dubbed_16k.wav + final_mix_aac.m4a> <work dir> <out.mp4> [route.json]
# exit 0 = OUT written; 3 = nothing to lip-sync (keep the source picture, not an error); other = failed.
# multi-shot (K2): KS_SHOTS=auto (default) -> ks_shots.py detect after the landmarks; > 1 shot -> ks_shots.sh runs this script once per shot (own
#   plan / crop / ref / KeySync / post / paste per shot, nothing smoothed across a cut), concat -c copy + the mix stream-copied. KS_SHOTS=off = one run.
# Env: KS_SHOTS KS_SHOT_MIN_S KS_DELAY KS_PROBE_S KS_REF KS_CROP KS_PAD KS_GATE KS_TSMOOTH KS_PAUSE KS_PASTE_SHIFT KS_PLAIN_ARGS KS_PLAIN_EXTRA KS_INTERP_STEPS KS_SYNC_MAX_S
#      KS_PASTE_JOBS KS_MIN_IOD KS_NUM_FACES KS_PROBE_MIN_CONF KS_PROBE_MAX KS_CHECK; GPU_RDY (gpulock.sh hand-off)
set -o pipefail
ID=$1; SRC=$2; AD=$3; W=$4; OUTV=$5; ROUTE=${6:-}
R=$(cd "$(dirname "$0")/../../.." && pwd); KS=${KS_ROOT:-/workspace/dub/keysync}; LOCK=${KS_GPU_LOCK:-/workspace/dub/gpu.lock}
PY=${DUB_PY:-/venv/dabai/bin/python}; MPY=${DUB_MEDIAPIPE_PY:-/venv/mediapipe/bin/python}; LORA=${KS_LORA:-$KS/student/lora_pick.pt}
# r16 T (Q audit O5): KS_PHASE=all (default) | prep | gpu | post. ks_shots.sh runs prep for every shot, then ONE GPU hold + ONE resident ks_fast for all
# shots' gpu phases (KS_FAST_FJ / KS_FAST_LOG / KS_FAST_OUT / KS_FAST_PID / KS_TAG_PREFIX), then the CPU post phases; the state lives in $W/ks.
PH=${KS_PHASE:-all}; KW=$W/ks
if [ "$PH" = all ] || [ "$PH" = prep ]; then rm -rf $KW; mkdir -p $KW/data $KW/out $KW/probe $KW/kit
  for f in $R/scripts/keysync/*.py $R/scripts/keysync/r15/*.py $R/scripts/keysync/r16/*.py; do cp $f $KW/kit/; done; fi   # one flat kit dir = this checkout's code (copies: python resolves a symlinked script's imports to its real dir)
KT=$KW/kit; PAD=${KS_PAD:-60}
flag() { echo "$1" >> $KW/flags.txt; echo "KS_FLAG $1 ${2:-}"; }
t() { echo "KS_T $1 $(date +%s.%N)"; }
dur() { ffprobe -v error -show_entries format=duration -of csv=p=0 "$1"; }
sync_off() {   # sync_off <video> <wav> <json>: SyncNet global AV offset of a raw KeySync render (first KS_SYNC_MAX_S s); prints the offset or "x"
  local V=$1 A=$2 J=$3 MX=${KS_SYNC_MAX_S:-60}
  if [ "$MX" != 0 ] && python3 -c "import sys; sys.exit(0 if float(sys.argv[1]) > float(sys.argv[2]) else 1)" "$(dur $V)" "$MX"; then
    ffmpeg -v error -y -i $V -t $MX -an -c:v copy ${J%.json}.mp4 && ffmpeg -v error -y -i $A -t $MX ${J%.json}.wav && V=${J%.json}.mp4 A=${J%.json}.wav; fi
  ( cd $R && $PY scripts/quality/sync_perframe.py --video $V --audio $A --out $J ) > ${J%.json}.log 2>&1 \
    && python3 -c "import json; print(int(json.load(open('$J'))['global']['av_offset']))" || echo x; }

if [ "$PH" = all ] || [ "$PH" = prep ]; then   # ==== prep (CPU): master, landmarks, shots, plan, crop, pad, reference
t master
if [ -n "${KS_MASTER_IN:-}" ]; then cp "$KS_MASTER_IN" $KW/master_25fps.mp4   # (ks_shots.sh: this shot's frames of the parent master, lossless)
else ffmpeg -v error -y -i "$SRC" -map 0:v:0 -an -vf fps=25 -c:v libx264 -preset fast -crf 10 -pix_fmt yuv420p $KW/master_25fps.mp4 || exit 1; fi
N=$(ffprobe -v error -count_packets -select_streams v:0 -show_entries stream=nb_read_packets -of csv=p=0 $KW/master_25fps.mp4)
t landmarks   # CPU; up to KS_NUM_FACES faces (route.json faces are matched to these tracks)
if [ -n "${KS_MP_NPZ:-}" ]; then cp "$KS_MP_NPZ" $KW/master_mp.npz   # (ks_shots.sh: the parent's landmarks sliced to this shot)
else GLOG_minloglevel=2 OMP_NUM_THREADS=4 $MPY $R/scripts/pilot/mp_landmarks.py $KW/master_25fps.mp4 $KW/master_mp.npz --num-faces ${KS_NUM_FACES:-4} > $KW/mp.log 2>&1 || { tail -3 $KW/mp.log; exit 1; }; fi
if [ "${KS_SHOTS:-auto}" != off ] && [ -z "${KS_IN_SHOT:-}" ]; then   # K2 multi-shot: > 1 segment (hard cuts + faceless gaps > 2 s, each cut to its face range) -> ks_shots.sh runs this script per face segment
  t shots; $PY $KT/ks_shots.py detect --video $KW/master_25fps.mp4 --mp $KW/master_mp.npz --out $KW/shots.json | cut -c1-600 || exit 1
  if [ "$(python3 -c "import json; print(len(json.load(open('$KW/shots.json'))['segments']))")" -gt 1 ]; then bash $R/scripts/keysync/r16/ks_shots.sh "$ID" "$AD" "$W" "$OUTV" "$ROUTE" $KW; exit $?; fi
fi
t plan; $PY $R/scripts/keysync/r16/ks_e2e_prep.py --mp $KW/master_mp.npz --dub $AD/dubbed_16k.wav --out $KW ${ROUTE:+--route $ROUTE} | tee $KW/plan.log | cut -c1-600 || exit 1
TAGS=$(python3 -c "import json; print(' '.join(f['tag'] for f in json.load(open('$KW/faces.json'))['faces']))")
[ -n "$TAGS" ] || { echo "KS_FACE_KEEP_SOURCE $ID: no face to lip-sync ($(python3 -c "import json; d=json.load(open('$KW/faces.json')); print(d['mode'], d.get('skipped'))"))"; exit 3; }

t crop
for f in $TAGS; do
  # the r15 static crop (ks_prep rule; another face in the shot: the tight 1.25 / 0.4 box of r15 K m03, so KeySync / SyncNet see one face)
  BK=""; [ -s $KW/others_$f.npz ] && BK="KS_BOX_K=${KS_BOX_K:-1.25} KS_BOX_M=${KS_BOX_M:-0.4}"
  ( mkdir -p $KW/data/st/videos $KW/data/st/audios; cd $KW && env $BK $PY $KT/ks_prep.py $KW/master_25fps.mp4 $KW/lm_$f.npz $KW/dub_$f.wav $KW/data/st $f ) > $KW/crop_$f.log 2>&1 || { tail -3 $KW/crop_$f.log; exit 1; }
  CR=${KS_CROP:-auto}
  if [ "$CR" = auto ]; then   # K2 rule: the tracked crop when the static face share < 0.45 (q05 small face) or the head travels > 0.4 face heights (q02 blob)
    CR=$(python3 -c "import json; b=json.load(open('$KW/data/st/box_$f.json')); print('track' if b['face_h'] / b['side'] < 0.45 or b['move'] / b['face_h'] > 0.4 else 'static')"); fi
  echo "KS_CROP $f $CR $(python3 -c "import json; b=json.load(open('$KW/data/st/box_$f.json')); print('static face share %.2f, head travel %.2f face heights' % (b['face_h'] / b['side'], b['move'] / b['face_h']))")"
  if [ "$CR" = track ]; then   # K2: constant side, the centre follows the face (smoothed)
    TK=${KS_TRACK_K:-1.8}; [ -s $KW/others_$f.npz ] && TK=${KS_TRACK_K2:-1.25}
    $MPY $KT/k2_prep_track.py --master $KW/master_25fps.mp4 --lm $KW/lm_$f.npz --out $KW/data/crop_$f --k $TK >> $KW/crop_$f.log 2>&1 || { tail -3 $KW/crop_$f.log; exit 1; }
    mv $KW/data/crop_${f}_box.json $KW/data/box_$f.json
  else
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
  REF=${KS_REF:-0}   # Irina 08.11 (m18 clip): ref 0 + the gate = the default; KS_REF=auto = K2 refpick (option)
  if [ "$REF" = auto ]; then   # K2: a frame with the upper teeth showing as KeySync's identity (black-hole mouths roughly halved); 0 if the pick fails
    REF=$($MPY $KT/k2_refpick.py --video $KW/data/crop_$f.mp4 --lm $KW/crop_lm_$f.npz --face-lm $KW/lm_$f.npz --box $KW/data/box_$f.json --out $KW/ref_$f 2> $KW/ref_$f.log | grep "^K2_REFPICK" | python3 -c "import json,sys; print(json.loads(sys.stdin.read()[11:])['pick'])" 2>/dev/null)
    [ -n "$REF" ] || { echo "KS WARN refpick failed ($f): ref 0"; REF=0; }
  fi
  echo $REF > $KW/ref_$f.txt
done
fi   # ==== end of prep
[ "$PH" = prep ] && { echo "KS_PHASE_DONE prep $ID"; exit 0; }
N=$(ffprobe -v error -count_packets -select_streams v:0 -show_entries stream=nb_read_packets -of csv=p=0 $KW/master_25fps.mp4)
TAGS=$(python3 -c "import json; print(' '.join(f['tag'] for f in json.load(open('$KW/faces.json'))['faces']))")
declare -A DLY SHF
if [ "$PH" = all ] || [ "$PH" = gpu ]; then   # ==== gpu
# ---- GPU section = ONE hold of the GPU lock (engines, probe, full KeySync, SyncNet checks); released before the CPU post / paste.
# GPU_RDY (scripts/keysync/r16/gpulock.sh): the caller (e2e.sh with R10_HEAVY_LOCK) already holds the GPU lock since its audio stage -> no second
# wait in the queue; released here (gpu_release) before the CPU post. Without it this script takes the lock itself. No lock fd is ever inherited.
. $R/scripts/keysync/r16/gpulock.sh
SHR=${KS_FAST_FJ:-}   # shared resident ks_fast (ks_shots.sh holds the GPU lock for all shots)
[ -n "$SHR" ] || gpu_held || { t wait_gpu; gpu_acquire $$ || { echo "KS: could not take the GPU lock"; exit 1; }; }
t engines; ENG=$(KS_GPU_LOCK=/dev/null bash $R/scripts/keysync/r16/ks_engines.sh 2> $KW/engines.log | tail -1); [ -d "$ENG" ] || { tail -5 $KW/engines.log; exit 1; }
LMK=${KS_LMK_CACHE:-$KS/lmk}; mkdir -p $LMK; [ -n "${KS_TORCH_HOME:-}" ] && export TORCH_HOME=$KS_TORCH_HOME   # (image: face_alignment's weights on the volume)
GATE="${KS_GATE-thr=0.12,cap=0.25,hi=8,lo=2}"
ksfast() {   # ksfast <jobs> <out> <log>: the resident KeySync-fast worker (fast2: TRT keyframes + 2-step LCM student + the K2 gate, compiled VAE decode)
  ( cd $KS/code && . ../venv/bin/activate && env KS_DC=2:0 KS_LMK_CACHE=$LMK TORCHINDUCTOR_CACHE_DIR=${TORCHINDUCTOR_CACHE_DIR:-$KS/inductor} ${GATE:+KS_GATE=$GATE} \
    python $KT/ks_fast.py --jobs $1 --out $2 --cfg _s8 --vae-half --compile unet --trt $ENG --interp-lora $LORA --interp-steps ${KS_INTERP_STEPS:-2} \
      --vae-batch 8 --vae-compile ) > $3 2>&1 || { tail -5 $3; return 1; }; grep "^KS_JOB" $3 | cut -c1-400; }
if [ -n "$SHR" ]; then FJ=$SHR; KLOG=$KS_FAST_LOG; KOUT=$KS_FAST_OUT; KSP=$KS_FAST_PID; PFX=${KS_TAG_PREFIX:-}
else FJ=$KW/jobs_follow.txt; : > $FJ; KLOG=$KW/ks_fast.log; KOUT=$KW/out; PFX=""
( cd $KS/code && . ../venv/bin/activate && env KS_DC=2:0 KS_LMK_CACHE=$LMK TORCHINDUCTOR_CACHE_DIR=${TORCHINDUCTOR_CACHE_DIR:-$KS/inductor} ${GATE:+KS_GATE=$GATE} \
  python $KT/ks_fast.py --follow --jobs $FJ --out $KW/out --cfg _s8 --vae-half --compile unet --trt $ENG --interp-lora $LORA --interp-steps ${KS_INTERP_STEPS:-2} \
    --vae-batch 8 --vae-compile ) > $KW/ks_fast.log 2>&1 & KSP=$!
trap '[ -n "${KSP:-}" ] && kill $KSP 2>/dev/null' EXIT; fi   # never leave a --follow ks_fast behind (it would wait for END forever)
waitjob() { until grep -q "^KS_JOB {\"tag\": \"$1\"" $KLOG; do kill -0 $KSP 2>/dev/null || { echo "KS: ks_fast died before job $1"; tail -5 $KLOG; return 1; }; sleep 0.5; done; }
if [ "${KS_DELAY:-probe}" = probe ]; then
  t probe; : > $KW/probe/jobs.txt; PS=${KS_PROBE_S:-10}; OFFS=""
  for f in $TAGS; do   # a FRAME-ALIGNED window of KS_PROBE_S s from the face's first own speech; the same reference (frame 0) + gate as the full run
    $MPY - $KW/dub_$f.wav $KW/data/crop_$f.mp4 $(cat $KW/ref_$f.txt) $PS $KW/probe/v_$f.mp4 $KW/probe/a_$f.wav <<'PY' || exit 1
import subprocess, sys, wave, numpy as np, cv2
wav, crop, ref, ps, ov, oa = sys.argv[1], sys.argv[2], int(sys.argv[3]), float(sys.argv[4]), sys.argv[5], sys.argv[6]
w = wave.open(wav); sr = w.getframerate(); x = np.frombuffer(w.readframes(w.getnframes()), np.int16); spf = sr // 25
e = np.sqrt(np.convolve(x.astype(np.float32) ** 2, np.ones(sr // 10) / (sr // 10), "same")); on = np.flatnonzero(e > 0.05 * (e.max() + 1e-6))
c = cv2.VideoCapture(crop); n = int(c.get(cv2.CAP_PROP_FRAME_COUNT)); L = int(ps * 25)
k0 = int(max(0, min(round(((on[0] / sr) - 0.5) * 25) if len(on) else 0, n - L)))   # whole frames: video frame k0 <-> audio sample k0 * spf exactly
c.set(cv2.CAP_PROP_POS_FRAMES, ref); fr = c.read()[1]; c.set(cv2.CAP_PROP_POS_FRAMES, k0); F = [fr]
for _ in range(L):
    ok, f = c.read()
    if not ok: break
    F.append(f)
p = subprocess.Popen(["ffmpeg", "-v", "error", "-y", "-f", "rawvideo", "-pix_fmt", "bgr24", "-s", "512x512", "-r", "25", "-i", "-", "-c:v", "libx264", "-crf", "10", "-pix_fmt", "yuv420p", ov], stdin=subprocess.PIPE)
for f in F: p.stdin.write(f.tobytes())
p.stdin.close(); p.wait()
seg = np.concatenate([np.zeros(spf, np.int16), x[k0 * spf:(k0 + len(F) - 1) * spf]])   # frame 0 = the reference <-> one frame of silence
with wave.open(oa, "wb") as o: o.setnchannels(1); o.setsampwidth(2); o.setframerate(sr); o.writeframes(seg.tobytes())
print(f"probe window frames {k0}..{k0 + len(F) - 1} (+ ref {ref} as frame 0)", file=sys.stderr)
PY
    echo "${PFX}p_$f $KW/probe/v_$f.mp4 $KW/probe/a_$f.wav 0" >> $FJ
  done
  for f in $TAGS; do waitjob ${PFX}p_$f || exit 1; done; grep "^KS_JOB {\"tag\": \"${PFX}p_" $KLOG | cut -c1-400
  for f in $TAGS; do   # delay = -offset x 40 ms; |offset| clamped to KS_PROBE_MAX (2) frames; SyncNet conf < KS_PROBE_MIN_CONF (3) or no result -> 0 + a flag
    O=$(sync_off $(ls $KOUT/${PFX}p_$f/*.mp4 | head -1) $KW/probe/a_$f.wav $KW/probe/sync_$f.json)
    C=$(python3 -c "import json; print(json.load(open('$KW/probe/sync_$f.json'))['global']['conf'])" 2>/dev/null || echo 0)
    if [ "$O" = x ]; then flag ks_probe_failed "$f: no SyncNet result -> delay 0"; O=0
    elif python3 -c "import sys; sys.exit(0 if float('$C') < float('${KS_OFFSCREEN_CONF:-1.5}') else 1)"; then   # r16c T (Irina 09.11 rule 2): this face does NOT speak the dub
      flag ks_offscreen_voice "$f: probe conf $C < ${KS_OFFSCREEN_CONF:-1.5}: the voice is not this face's (off-screen) -> its source picture"; OFFS="$OFFS $f"; O=0   # (q02 0.08; every real speaker >= 3.4)
    elif python3 -c "import sys; sys.exit(0 if float('$C') < float('${KS_PROBE_MIN_CONF:-3}') else 1)"; then flag ks_probe_low_conf "$f: conf $C, offset $O -> delay 0"; O=0
    elif [ "${O#-}" -gt "${KS_PROBE_MAX:-2}" ]; then flag ks_probe_clamped "$f: offset $O -> +-${KS_PROBE_MAX:-2}"; O=$(( O > 0 ? ${KS_PROBE_MAX:-2} : -${KS_PROBE_MAX:-2} )); fi
    DLY[$f]=$(( -O * 40 )); echo "KS_PROBE $f av_offset $O conf $C -> dub delay ${DLY[$f]} ms"; done
  if [ -n "$OFFS" ]; then TAGS=$(for f in $TAGS; do [[ " $OFFS " == *" $f "* ]] || echo -n "$f "; done)   # the off-screen-voice faces get no render
    if [ -z "${TAGS// }" ]; then echo "KS_FACE_KEEP_SOURCE $ID: off-screen voice (faces$OFFS): the source picture with the dub"
      [ -n "$SHR" ] || { echo END >> $FJ; wait $KSP 2>/dev/null; KSP=""; gpu_release; }; exit 3; fi; fi
else for f in $TAGS; do DLY[$f]=${KS_DELAY}; done; fi
t keysync; : > $KW/jobs.txt
for f in $TAGS; do   # the dub KeySync hears: delayed (or advanced) by the probe, padded with silence to the padded crop (the final mix is never touched)
  D=${DLY[$f]}; PD=$(python3 -c "print(($N + $PAD) / 25)")
  if [ "$D" -ge 0 ]; then AF="adelay=$D:all=1,apad=whole_dur=$PD"; else AF="atrim=start=$(python3 -c "print(-$D / 1000)"),asetpts=PTS-STARTPTS,apad=whole_dur=$PD"; fi
  ffmpeg -v error -y -i $KW/dub_$f.wav -af "$AF" -t $PD -ar 16000 -ac 1 $KW/data/dubp_$f.wav || exit 1
  ffmpeg -v error -y -i $KW/dub_$f.wav -af "apad=whole_dur=$PD" -t $PD -ar 16000 -ac 1 $KW/data/dubo_$f.wav || exit 1   # the ORIGINAL timing, for the check
  if [ "$N" -gt "$(( ${KS_CHUNK_MIN_S:-90} * 25 ))" ]; then   # long video: ~KS_CHUNK_S pieces cut at dub pauses, the same reference in each (RAM flat)
    $PY $KT/ks_chunk.py split --video $KW/data/pad_$f.mp4 --audio $KW/data/dubp_$f.wav --ref $(cat $KW/ref_$f.txt) --chunk-s ${KS_CHUNK_S:-60} --out $KW/data/chunks_$f --tag $PFX$f \
      >> $KW/jobs.txt 2> $KW/chunk_$f.log || { tail -3 $KW/chunk_$f.log; exit 1; }; grep KS_CHUNKS $KW/chunk_$f.log
  else echo "$PFX$f $KW/data/pad_$f.mp4 $KW/data/dubp_$f.wav $(cat $KW/ref_$f.txt)" >> $KW/jobs.txt; fi
done
cat $KW/jobs.txt >> $FJ
if [ -n "$SHR" ]; then waitjob "$(tail -1 $KW/jobs.txt | awk '{print $1}')" || exit 1   # (jobs run in order: the last one done = all done)
else echo END >> $FJ; wait $KSP || { tail -5 $KLOG; exit 1; }; fi
grep "^KS_JOB" $KLOG | grep -v "\"tag\": \"${PFX}p_" | grep "\"tag\": \"$PFX" | cut -c1-400
for f in $TAGS; do
  if [ -d $KW/data/chunks_$f ]; then rm -rf $KW/out/$f; mkdir -p $KW/out/$f
    $PY $KT/ks_chunk.py join --dir $KW/data/chunks_$f --renders $KOUT --out $KW/out/$f/joined.mp4 --tag $PFX$f || exit 1
  elif [ -n "$SHR" ]; then rm -rf $KW/out/$f; ln -sfn $KOUT/$PFX$f $KW/out/$f; fi; done
t check; declare -A SHF
for f in $TAGS; do   # the raw render vs the ORIGINAL dub (K2): ~0 after the probe delay. KS_PASTE_SHIFT: 0 (default, the probe only) | auto (= -offset) | N
  SH=${KS_PASTE_SHIFT:-0}; O=x
  if [ "${KS_CHECK:-on}" = on ] || [ "$SH" = auto ]; then O=$(sync_off $(ls $KW/out/$f/*.mp4 | head -1) $KW/data/dubo_$f.wav $KW/sync_$f.json); fi
  [ "$O" = x ] && [ "${KS_CHECK:-on}" = on ] && flag ks_check_failed "$f: the post-render SyncNet check gave no result"   # (Q audit O9)
  [ "$O" != x ] && [ "${O#-}" -ge 2 ] && flag ks_check_offset "$f: raw render vs the original dub still $O frames after the probe delay"
  if [ "$SH" = auto ]; then [ "$O" = x ] && O=0; SH=$(( -O )); fi
  SHF[$f]=$SH; echo "KS_CHECK $f delay ${DLY[$f]} ms, raw offset vs the original dub $O, paste shift $SH"
done
[ -n "$SHR" ] || gpu_release   # GPU done: the post + paste below are CPU only
declare -p DLY SHF > $KW/state.sh; echo "OFFS='${OFFS:-}'" >> $KW/state.sh
fi   # ==== end of gpu
[ "$PH" = gpu ] && { echo "KS_PHASE_DONE gpu $ID"; exit 0; }
[ "$PH" = post ] && . $KW/state.sh   # ==== post (CPU)
[ -n "${OFFS:-}" ] && TAGS=$(for f in $TAGS; do [[ " $OFFS " == *" $f "* ]] || echo -n "$f "; done)   # (r16c T: no post / paste for an off-screen-voice face)

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
  if [ "${KS_SEAM:-on}" = on ]; then   # K1 r16-paste2: KeySync's 13-frame segment-boundary jumps spread over 3 transitions (per KeySync run: ks_chunk chunks)
    CJ=""; [ -s $KW/data/chunks_$f/chunks.json ] && CJ="--chunks $KW/data/chunks_$f/chunks.json"
    $PY $KT/ks_seam.py --layer $L --src $KW/data/pad_$f.mp4 --out $P/seam.mp4 $CJ > $P/seam.log 2>&1 && L=$P/seam.mp4 || { tail -3 $P/seam.log; exit 1; }; tail -1 $P/seam.log | cut -c1-200; fi
  if [ "${KS_PAUSE:-off}" = warp ]; then   # K2 pause (b), default OFF until Irina picks it: the lips closed in the dub's pauses (the layer is on the dub timeline: shift 0)
    $MPY $KT/k2_pause.py --mode warp --ks $L --ks-lm $P/raw_lm.npz --wav $KW/dub_$f.wav --shift 0 --out $P/pause.mp4 > $P/pause.log 2>&1 && L=$P/pause.mp4 || { tail -3 $P/pause.log; exit 1; }; fi
  t paste_$f; n=$((n + 1)); NXT=$KW/paste_$n.mp4
  OTH=""; [ -s $KW/others_$f.npz ] && OTH="--others $KW/others_$f.npz --others-grow ${KS_OTHERS_GROW:-1.3}"
  ARGS="--ks $L --ks-lm $P/raw_lm.npz --master $CUR --box $KW/data/box_$f.json --lm $KW/lm_$f.npz --audio $AD/final_mix_aac.m4a --shift ${SHF[$f]} --absent skip $OTH \
        ${KS_PLAIN_ARGS---lfm 6 --upper-src 0.12:0.32 --keep-nose 0.04 --align 2 --align-ecc $P/ecc.npz --align-scale 0 --tex 2} ${KS_PLAIN_EXTRA:-}"
  if [ "${KS_PASTE_JOBS:-0}" != 1 ] && [ -f $KT/ks_plain_par.py ]; then
    ( cd $KT && KS_CMS=1 $PY $KT/ks_plain_par.py --jobs ${KS_PASTE_JOBS:-$(( $(nproc) / 3 ))} --out $NXT -- $ARGS ) > $P/plain.log 2>&1 || {   # a failed parallel paste -> the serial one (== by K1 a6ded57)
      flag ks_paste_par_retry "$f: parallel paste failed ($(grep -a -E "Error" $P/plain.log | tail -1 | cut -c1-120)) -> serial"; mv $P/plain.log $P/plain_par_failed.log
      ( cd $KT && KS_CMS=1 $PY $KT/ks_plain.py $ARGS --out $NXT ) > $P/plain.log 2>&1 || { tail -5 $P/plain.log; exit 1; }; }
  else ( cd $KT && KS_CMS=1 $PY $KT/ks_plain.py $ARGS --out $NXT ) > $P/plain.log 2>&1 || { tail -5 $P/plain.log; exit 1; }; fi
  echo "KS_PASTE $f delay ${DLY[$f]} ms shift ${SHF[$f]} ref $(cat $KW/ref_$f.txt) $(grep -E 'plain paste:' $P/plain.log | head -1)"; CUR=$NXT
done
t end; mv $CUR $OUTV && echo "KS_FACE_DONE $ID faces $TAGS"
