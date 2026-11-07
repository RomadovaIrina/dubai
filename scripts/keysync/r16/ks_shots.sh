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
# KS_HELD_FD (e2e holds the GPU lock): the first ks_face run releases it; later segments take KS_GPU_LOCK themselves.
# exit: 0 = OUT written; 3 = nothing lip-synced (keep the source); other = failed.
set -o pipefail
ID=$1; AD=$2; W=$3; OUTV=$4; ROUTE=$5; PK=$6
R=$(cd "$(dirname "$0")/../../.." && pwd); PY=${DUB_PY:-/venv/dabai/bin/python}; KT=$PK/kit
SW=$W/shots; rm -rf $SW; mkdir -p $SW
N=$(python3 -c "import json; print(json.load(open('$PK/shots.json'))['frames'])")
SEGS=$(python3 -c "import json; print(json.dumps(json.load(open('$PK/shots.json'))['segments']))")
mapfile -t FS < <(python3 -c "
import json; s = json.loads('$SEGS')
for i, (k0, k1, kd) in sorted(enumerate(s), key=lambda x: -(x[1][1] - x[1][0])):
    if kd == 'face': print(i, k0, k1)")
NS=$(python3 -c "import json; print(len(json.loads('$SEGS')))"); echo "KS_SHOTS_RUN $ID: $NS segments, ${#FS[@]} with a face (longest first: ${FS[*]})"
if [[ "${KS_REF:-auto}" =~ ^[0-9]+$ ]]; then echo "KS WARN KS_REF=$KS_REF is a frame of the whole video: auto per shot"; export KS_REF=auto; fi
DELAY=${KS_DELAY:-probe}; MIN=$(python3 -c "print(round(${KS_SHOT_MIN_S:-1.0} * 25))"); LS=0; declare -A PART
for row in "${FS[@]}"; do
  read -r i k0 k1 <<< "$row"; S=$SW/s$i; mkdir -p $S/audio; t0=$(python3 -c "print($k0 / 25)"); t1=$(python3 -c "print($k1 / 25)")
  if [ $((k1 - k0)) -lt $MIN ]; then echo "KS_SHOT $i [$k0, $k1) $t0-$t1 s: shorter than ${KS_SHOT_MIN_S:-1.0} s -> source"; continue; fi
  ffmpeg -v error -y -i $PK/master_25fps.mp4 -vf "select=between(n\,$k0\,$((k1 - 1))),setpts=N/25/TB" -r 25 -c:v libx264 -qp 0 -preset ultrafast -pix_fmt yuv420p $S/master.mp4 || exit 1
  $PY $KT/ks_shots.py slice --mp $PK/master_mp.npz --k0 $k0 --k1 $k1 --out $S/master_mp.npz || exit 1
  ffmpeg -v error -y -i $AD/dubbed_16k.wav -af "atrim=start_sample=$((k0 * 640)):end_sample=$((k1 * 640)),asetpts=PTS-STARTPTS,apad=whole_len=$(((k1 - k0) * 640))" -ar 16000 -ac 1 $S/audio/dubbed_16k.wav || exit 1
  ffmpeg -v error -y -i $AD/final_mix_aac.m4a -af "atrim=start=$t0:end=$t1,asetpts=PTS-STARTPTS,apad=whole_dur=$(python3 -c "print($t1 - $t0 + 0.2)")" -c:a aac -b:a 192k $S/audio/final_mix_aac.m4a || exit 1
  for m in $AD/*.manifest.json; do [ -f "$m" ] && $PY $KT/ks_shots.py manifest --manifest $m --t0 $t0 --t1 $t1 --out $S/audio/$(basename $m) > /dev/null; done
  RS=""; if [ -n "$ROUTE" ] && [ -f "$ROUTE" ]; then $PY $KT/ks_shots.py route --route $ROUTE --t0 $t0 --t1 $t1 --out $S/route.json | sed "s/^/KS_SHOT $i /"; RS=$S/route.json; fi
  env KS_IN_SHOT=1 KS_MASTER_IN=$S/master.mp4 KS_MP_NPZ=$S/master_mp.npz KS_DELAY=$DELAY bash $R/scripts/keysync/r16/ks_face.sh ${ID}_s$i $S/master.mp4 $S/audio $S $S/out.mp4 $RS > $S/face.log 2>&1
  rc=$?; grep -E "^KS_(CROP|PROBE|CHECK|PASTE|FACE_KEEP|FACE_DONE)|WARN" $S/face.log | cut -c1-300 | sed "s/^/KS_SHOT $i [$k0, $k1) /"
  [ -n "${KS_HELD_FD:-}" ] && grep -q "^KS_T engines" $S/face.log && unset KS_HELD_FD   # (the first run's GPU section released it)
  if [ $rc = 3 ]; then echo "KS_SHOT $i [$k0, $k1) $t0-$t1 s: nothing to lip-sync -> source"; continue; fi
  [ $rc = 0 ] || { echo "KS_SHOT $i FAILED (rc $rc)"; tail -5 $S/face.log; exit 1; }
  if [ "$DELAY" = probe ] && ! grep -q "probe sync failed" $S/face.log; then
    P=$(grep -o "KS_PROBE f[0-9]* av_offset [-0-9]* -> dub delay [-0-9]* ms" $S/face.log | head -1 | awk '{print $(NF-1)}')
    [ -n "$P" ] && { DELAY=$P; echo "KS_SHOTS delay $DELAY ms (probed in segment $i) for the remaining segments"; }; fi
  PART[$i]=$S/out.mp4; LS=$((LS + 1))
done
[ $LS -gt 0 ] || { echo "KS_FACE_KEEP_SOURCE $ID: nothing lip-synced in $NS segments"; exit 3; }
L=(); for i in $(seq 0 $((NS - 1))); do   # in time order; the source segments re-encoded like ks_plain's output (concat -c copy needs the same stream)
  if [ -n "${PART[$i]:-}" ]; then L+=(${PART[$i]}); else
    read -r k0 k1 <<< "$(python3 -c "import json; s = json.loads('$SEGS')[$i]; print(s[0], s[1])")"
    ( cd $KT && KS_CMS=1 $PY $KT/ks_shots.py encode --video $PK/master_25fps.mp4 --k0 $k0 --k1 $k1 --out $SW/src_$i.mp4 ) || exit 1; L+=($SW/src_$i.mp4); fi; done
( cd $KT && $PY $KT/ks_shots.py concat --list ${L[@]} --audio $AD/final_mix_aac.m4a --frames $N --out $OUTV ) || exit 1
echo "KS_FACE_DONE $ID segments $NS lip-synced $LS"
