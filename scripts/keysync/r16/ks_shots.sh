#!/bin/bash
# r16 K2: the multi-shot KeySync face stage. Called by ks_face.sh (KS_SHOTS=auto, the default) when ks_shots.py detect returns > 1 segment:
# shots (hard cuts + faceless gaps > 2 s), each cut to its face range (KeySync cannot start on a faceless frame). ks_face.sh runs ONCE PER FACE
# SEGMENT, so the plan (route faces / tracks), the crop (static box / K2 tracked box + its smoothing), the reference frame (k2_refpick: own ref
# per shot), the landmark bridge, the KeySync job, the post (ECC, tsmooth, snap, guard) and the paste (lfm, align, colour match) never cross a
# cut (audit orch. 5, paste 12, prep 4). The outputs are concatenated with -c:v copy (source segments re-encoded exactly like ks_plain's output)
# and the FINAL MIX stream is COPIED (never re-encoded, no -shortest).
#   ks_shots.sh <id> <audio dir> <work dir> <out.mp4> <route.json or ""> <parent ks dir: master_25fps.mp4, master_mp.npz, shots.json, kit/>
# Per face segment: its master frames (lossless cut of ks_face's master: no extra generation), the MediaPipe npz sliced (KS_MP_NPZ: no 2nd
# landmark pass), the dub wav cut by samples, the mix cut (only for ks_plain's mux; dropped at the concat), route.json / manifest cut + shifted.
# Face segments run LONGEST FIRST: the first one whose delay probe SUCCEEDS sets KS_DELAY for the rest (one delay per video; no 10 s probe on
# short shots). A face segment < KS_SHOT_MIN_S (1.0 s) or with nothing to lip-sync (ks_face exit 3) keeps the source picture.
# GPU_RDY (e2e holds the GPU lock, gpulock.sh): the first ks_face run releases it; later segments take the lock themselves.
# exit: 0 = OUT written; 3 = nothing lip-synced (keep the source); other = failed.
set -o pipefail
ID=$1; AD=$2; W=$3; OUTV=$4; ROUTE=$5; PK=$6
R=$(cd "$(dirname "$0")/../../.." && pwd); PY=${DUB_PY:-/venv/dabai/bin/python}; KT=$PK/kit
SW=$W/shots; [ "${KS_SHOTS_REPOST:-0}" = 1 ] || rm -rf $SW; mkdir -p $SW   # (r16d T: a re-paste keeps the shots and their renders)
N=$(python3 -c "import json; print(json.load(open('$PK/shots.json'))['frames'])")
SEGS=$(python3 -c "import json; print(json.dumps(json.load(open('$PK/shots.json'))['segments']))")
mapfile -t FS < <(python3 -c "
import json; s = json.loads('$SEGS')
for i, (k0, k1, kd) in sorted(enumerate(s), key=lambda x: -(x[1][1] - x[1][0])):
    if kd == 'face': print(i, k0, k1)")
NS=$(python3 -c "import json; print(len(json.loads('$SEGS')))"); echo "KS_SHOTS_RUN $ID: $NS segments, ${#FS[@]} with a face (longest first: ${FS[*]})"
if [[ "${KS_REF:-0}" =~ ^[0-9]+$ ]] && [ "${KS_REF:-0}" != 0 ]; then echo "KS WARN KS_REF=$KS_REF is a frame of the whole video: 0 (the first frame) per shot"; export KS_REF=0; fi   # KS_REF=0 = each shot's first frame
DELAY=${KS_DELAY:-probe}; MIN=$(python3 -c "print(round(${KS_SHOT_MIN_S:-1.0} * 25))"); LS=0; declare -A PART ALPHA SHOTARGS SPAN; READY=(); NFAIL=0; FAILF=()
shot_log() { grep -h -E "^KS_(CROP|PROBE|CHECK|PASTE|FACE_KEEP|FACE_DONE|FLAG)|WARN" $SW/s$1/face_$2.log 2>/dev/null | cut -c1-300 | sed "s/^/KS_SHOT $1 ${SPAN[$1]} /"; }
shot_fail() {   # r16 T (Q audit O5): ONE failed segment keeps ITS source picture (flagged); the rest of the video is still lip-synced
  echo "KS_SHOT $1 ${SPAN[$1]} FAILED in $2 (${3:-}) -> source"; tail -3 $SW/s$1/face_$2.log 2>/dev/null; echo ks_shot_failed >> $PK/flags.txt; unset "ALPHA[$1]"; NFAIL=$((NFAIL + 1))
  FAILF+=("$(echo "${SPAN[$1]}" | tr -d '[),' | awk '{print $2 - $1}')"); }   # r16 TZ 11.2: the source frames a failed shot keeps
for row in "${FS[@]}"; do
  read -r i k0 k1 <<< "$row"; S=$SW/s$i; mkdir -p $S/audio
  if [ $((k1 - k0)) -lt $MIN ]; then echo "KS_SHOT $i [$k0, $k1): shorter than ${KS_SHOT_MIN_S:-1.0} s -> source"; continue; fi
  ZS=(); if [ "${KS_DISSOLVE:-auto}" = auto ]; then   # r16 T + K2 ks_dissolve.py: alpha-paste through clean linear crossfades at this segment's edges
    while read -r d0 d1 tf bf; do [ -n "$d0" ] || continue
      $PY $KT/ks_dissolve.py check --master $PK/master_25fps.mp4 --d0 $d0 --d1 $d1 --t-frame $tf --b-frame $bf --out $S/dissolve_$d0.json 2>/dev/null | grep -E "^KS_(DISSOLVE_CHECK|WARN)" | sed "s/^/KS_SHOT $i /"
      if python3 -c "import json,sys; sys.exit(0 if json.load(open('$S/dissolve_$d0.json'))['ok'] else 1)" 2>/dev/null; then ZS+=("$d0:$d1:$tf:$bf")
      else echo "KS_FLAG ks_dissolve_hard_switch segment $i frames $d0-$d1"; echo ks_dissolve_hard_switch >> $PK/flags.txt; fi
    done < <(python3 -c "import json; [print(t['d0'], t['d1'], t['t_frame'], t['b_frame']) for t in json.load(open('$PK/shots.json')).get('transitions', []) if t['seg'] == [$k0, $k1]]")
  fi
  if [ ${#ZS[@]} -gt 0 ]; then   # extend the segment over the clean zones; the KeySync input = the unmixed face (ks_alpha.py); its own landmark pass
    for z in "${ZS[@]}"; do IFS=: read -r d0 d1 _ _ <<< "$z"; [ $d0 -lt $k0 ] && k0=$d0; [ $d1 -gt $k1 ] && k1=$d1; done
    $PY $KT/ks_alpha.py --master $PK/master_25fps.mp4 --k0 $k0 --k1 $k1 $(for z in "${ZS[@]}"; do echo --zone $z; done) --out $S/master.mp4 | sed "s/^/KS_SHOT $i /" || exit 1
    MPX=""; ALPHA[$i]="$k0"
  else
    ffmpeg -v error -y -i $PK/master_25fps.mp4 -vf "select=between(n\,$k0\,$((k1 - 1))),setpts=N/25/TB" -r 25 -c:v libx264 -qp 0 -preset ultrafast -pix_fmt yuv420p $S/master.mp4 || exit 1
    $PY $KT/ks_shots.py slice --mp $PK/master_mp.npz --k0 $k0 --k1 $k1 --out $S/master_mp.npz || exit 1; MPX=$S/master_mp.npz
  fi
  t0=$(python3 -c "print($k0 / 25)"); t1=$(python3 -c "print($k1 / 25)")
  ffmpeg -v error -y -i $AD/dubbed_16k.wav -af "atrim=start_sample=$((k0 * 640)):end_sample=$((k1 * 640)),asetpts=PTS-STARTPTS,apad=whole_len=$(((k1 - k0) * 640))" -ar 16000 -ac 1 $S/audio/dubbed_16k.wav || exit 1
  ffmpeg -v error -y -i $AD/final_mix_aac.m4a -af "atrim=start=$t0:end=$t1,asetpts=PTS-STARTPTS,apad=whole_dur=$(python3 -c "print($t1 - $t0 + 0.2)")" -c:a aac -b:a 192k $S/audio/final_mix_aac.m4a || exit 1
  for m in $AD/*.manifest.json; do [ -f "$m" ] && $PY $KT/ks_shots.py manifest --manifest $m --t0 $t0 --t1 $t1 --out $S/audio/$(basename $m) > /dev/null; done
  RS=""; if [ -n "$ROUTE" ] && [ -f "$ROUTE" ]; then $PY $KT/ks_shots.py route --route $ROUTE --t0 $t0 --t1 $t1 --out $S/route.json | sed "s/^/KS_SHOT $i /"; RS=$S/route.json; fi
  SHOTARGS[$i]="${ID}_s$i $S/master.mp4 $S/audio $S $S/out.mp4 $RS"; SPAN[$i]="[$k0, $k1)"
  if [ "${KS_FAULT:-}" = "shot:$i" ]; then shot_fail $i prep "KS_FAULT=shot:$i injected"; continue; fi   # r16 TZ 11.1 / 11.2 fault injection
  if [ "${KS_SHOTS_REPOST:-0}" = 1 ]; then [ -f $S/ks/state.sh ] && rc=0 || rc=3   # r16d T: re-paste only: the shot keeps its prep + KeySync render (state.sh = rendered)
  else env KS_IN_SHOT=1 KS_PHASE=prep KS_MASTER_IN=$S/master.mp4 KS_MP_NPZ=$MPX bash $R/scripts/keysync/r16/ks_face.sh ${SHOTARGS[$i]} > $S/face_prep.log 2>&1; rc=$?; fi
  shot_log $i prep
  if [ $rc = 3 ]; then echo "KS_SHOT $i ${SPAN[$i]}: nothing to lip-sync -> source"; unset "ALPHA[$i]"; continue; fi
  [ $rc = 0 ] || { shot_fail $i prep $rc; continue; }
  READY+=($i)
done
# ---- r16 T (Q audit O5): ONE GPU-lock hold + ONE resident ks_fast (one model / TRT load + VAE compile) for EVERY shot's probe + render
OKG=()
if [ "${KS_SHOTS_REPOST:-0}" = 1 ]; then OKG=("${READY[@]}"); echo "KS_SHOTS_REPOST: post + paste only for segments ${OKG[*]}"
elif [ ${#READY[@]} -gt 0 ]; then
  . $R/scripts/keysync/r16/gpulock.sh; gpu_held || { echo "KS_T wait_gpu $(date +%s.%N)"; gpu_acquire $$ || { echo "KS_SHOTS: could not take the GPU lock"; exit 1; }; }
  KS=${KS_ROOT:-/workspace/dub/keysync}; LORA=${KS_LORA:-$KS/student/lora_pick.pt}; LMK=${KS_LMK_CACHE:-$KS/lmk}; mkdir -p $LMK; [ -n "${KS_TORCH_HOME:-}" ] && export TORCH_HOME=$KS_TORCH_HOME
  GATE="${KS_GATE-thr=0.12,cap=0.25,hi=8,lo=2}"
  ENG=$(KS_GPU_LOCK=/dev/null bash $R/scripts/keysync/r16/ks_engines.sh 2> $SW/engines.log | tail -1); [ -d "$ENG" ] || { tail -5 $SW/engines.log; exit 1; }
  KFJ=$SW/jobs_follow.txt; : > $KFJ; KOUT=$SW/ksout; mkdir -p $KOUT
  . $R/scripts/keysync/r16/ks_serve.sh; ks_fast_launch $KFJ $KOUT $SW/ks_fast.log   # r17: resident KeySync daemon or a cold --follow ks_fast (ks_serve.sh)
  trap '[ -n "${KSP:-}" ] && kill $KSP 2>/dev/null' EXIT   # never leave the --follow ks_fast behind
  for i in "${READY[@]}"; do S=$SW/s$i
    if ! kill -0 $KSP 2>/dev/null; then shot_fail $i gpu "ks_fast died"; continue; fi
    env KS_IN_SHOT=1 KS_PHASE=gpu KS_DELAY=$DELAY KS_FAST_FJ=$KFJ KS_FAST_LOG=$SW/ks_fast.log KS_FAST_OUT=$KOUT KS_FAST_PID=$KSP KS_TAG_PREFIX=s${i}_ \
      bash $R/scripts/keysync/r16/ks_face.sh ${SHOTARGS[$i]} > $S/face_gpu.log 2>&1; rc=$?
    shot_log $i gpu
    if [ $rc = 3 ]; then echo "KS_SHOT $i ${SPAN[$i]}: off-screen voice -> source"; unset "ALPHA[$i]"; continue; fi   # r16c T: not a failure (flag ks_offscreen_voice)
    [ $rc = 0 ] || { shot_fail $i gpu $rc; continue; }
    if [ "$DELAY" = probe ] && ! grep -q -E "KS_FLAG ks_probe_(failed|low_conf)" $S/face_gpu.log; then   # only a CONFIDENT probe sets the video's delay
      P=$(grep -o -E "KS_PROBE f[0-9]+ av_offset [-0-9]+( conf [0-9.]+)? -> dub delay [-0-9]+ ms" $S/face_gpu.log | head -1 | awk '{print $(NF-1)}')
      [ -n "$P" ] && { DELAY=$P; echo "KS_SHOTS delay $DELAY ms (probed in segment $i) for the remaining segments"; }; fi
    OKG+=($i)
  done
  echo END >> $KFJ; wait $KSP 2>/dev/null; KSP=""; gpu_release
  grep -c "^KS_JOB" $SW/ks_fast.log | sed "s/^/KS_SHOTS one ks_fast: jobs /"
fi
# ---- the CPU post + paste of each shot (outside the GPU lock)
for i in "${OKG[@]}"; do S=$SW/s$i
  env KS_IN_SHOT=1 KS_PHASE=post bash $R/scripts/keysync/r16/ks_face.sh ${SHOTARGS[$i]} > $S/face_post.log 2>&1; rc=$?
  shot_log $i post
  [ $rc = 0 ] || { shot_fail $i post $rc; continue; }
  PART[$i]=$S/out.mp4; LS=$((LS + 1))
done
for i in "${!SHOTARGS[@]}"; do cat $SW/s$i/ks/flags.txt >> $PK/flags.txt 2>/dev/null; done   # the shots' flags reach the job status
python3 - $PK/subst.json "${FAILF[*]:-}" $SW/s*/ks/subst.json <<'PY'   # r16 TZ 11.2 / 11.3: the shots' substituted frames + the failed shots' source frames
import json, sys
out, failed = sys.argv[1], [int(x) for x in sys.argv[2].split()]; rows = [json.load(open(f)) for f in sys.argv[3:] if not f.endswith("*/ks/subst.json")]
d = {"eligible_frames": sum(r["eligible_frames"] for r in rows) + sum(failed), "guard_replaced": sum(r["guard_replaced"] for r in rows),
     "failed_frames": sum(failed), "failed_shots": len(failed), "shots": len(rows) + len(failed)}
json.dump(d, open(out, "w")); print("KS_SUBST", json.dumps(d))
PY
[ $LS -gt 0 ] || [ $NFAIL = 0 ] || { echo "KS_SHOTS: every lip-sync segment failed ($NFAIL)"; exit 1; }
[ $LS -gt 0 ] || { echo "KS_FACE_KEEP_SOURCE $ID: nothing lip-synced in $NS segments"; exit 3; }
L=(); for i in $(seq 0 $((NS - 1))); do   # in time order; the source segments re-encoded like ks_plain's output (concat -c copy needs the same stream)
  if [ -n "${PART[$i]:-}" ] && [ -z "${ALPHA[$i]:-}" ]; then L+=(${PART[$i]}); else
    read -r k0 k1 <<< "$(python3 -c "import json; s = json.loads('$SEGS')[$i]; print(s[0], s[1])")"
    ( cd $KT && KS_CMS=1 $PY $KT/ks_shots.py encode --video $PK/master_25fps.mp4 --k0 $k0 --k1 $k1 --out $SW/src_$i.mp4 ) || exit 1; L+=($SW/src_$i.mp4); fi; done
if [ ${#ALPHA[@]} -gt 0 ]; then   # the alpha-pasted segments: out = S + a (P - U) over the concat of the rest (K2 ks_dissolve assemble; the mix stream-copied)
  ( cd $KT && $PY $KT/ks_shots.py concat --list ${L[@]} --audio $AD/final_mix_aac.m4a --frames $N --out $SW/concat_base.mp4 ) || exit 1
  PA=(); for i in "${!ALPHA[@]}"; do PA+=("${PART[$i]}:${ALPHA[$i]}:$SW/s$i/master.mp4"); done
  ( cd $KT && KS_CMS=1 $PY $KT/ks_dissolve.py assemble --master $SW/concat_base.mp4 --audio $AD/final_mix_aac.m4a --part ${PA[@]} --out $OUTV ) | sed "s/^/KS_SHOTS /" || exit 1
else ( cd $KT && $PY $KT/ks_shots.py concat --list ${L[@]} --audio $AD/final_mix_aac.m4a --frames $N --out $OUTV ) || exit 1; fi
echo "KS_FACE_DONE $ID segments $NS lip-synced $LS"
