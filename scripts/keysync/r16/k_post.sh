#!/bin/bash
# r16 K1: the full post chain for ONE KeySync render (CPU only; run from /workspace/dub/keysync where ks_*.py / k_*.py live).
# usage: bash k_post.sh <work_dir> <ks_raw512.mp4> <src_crop512.mp4> <src_crop512.mediapipe.npz> <master_25fps.mp4> <box.json> <source_lm_F.npz> <final_mix.m4a> <dub16k.wav> <OUT.mp4> [faces2.npz]
#   ks_raw512   = KeySync output on the PADDED inputs (crop + 60 mirrored frames WITHOUT the duplicated last frame; dub = adelay(offset) + apad), longer than the master
#   src_crop512 = the same padded crop fed to KeySync (ECC + guard compare against it); src_crop512.mediapipe.npz = mediapipe of the UNPADDED source crop
#   faces2.npz  = k_faces2.py output for multi-face videos (enables the child protection)
# chain: mediapipe(raw) -> k_ecc -> ks_tsmooth gate:0.5 -> ks_snap -> ks_guard -> ks_plain (shift 0, ECC align scale-locked, lfm 6, upper-src, keep-nose[, others-grow 1.3]) -> SyncNet gate
set -e; W=$1; R=$2; SC=$3; SCL=$4; M=$5; B=$6; LF=$7; MIX=$8; DUB=$9; OUT=${10}; OTH=${11:-}; mkdir -p $W; K=$(pwd)
N=$(ffprobe -v error -count_frames -select_streams v:0 -show_entries stream=nb_read_frames -of csv=p=0 $M)
[ -s $W/raw.mp.npz ] || (cd /workspace/dub/dubai && /venv/mediapipe/bin/python scripts/quality/mp_landmarks.py --video $(realpath $R) --out $(realpath -m $W)/raw.mp.npz >/dev/null 2>&1 </dev/null)
/venv/dabai/bin/python k_ecc.py --ks $R --src $SC --lm $SCL --out $W/ecc.npz </dev/null
/venv/dabai/bin/python ks_tsmooth.py --layer $R --lm $W/raw.mp.npz --src-lm $SCL --out $W/ts.mp4 --alpha gate:0.5 </dev/null 2>&1 | grep gate:
/venv/dabai/bin/python ks_snap.py --layer $W/ts.mp4 --lm $W/raw.mp.npz --out $W/snap.mp4 </dev/null | grep ks_snap
/venv/dabai/bin/python ks_guard.py --layer $W/snap.mp4 --src $SC --n $N --out $W/guard.mp4 </dev/null
OA=""; [ -n "$OTH" ] && OA="--others $OTH --others-grow 1.3"
KS_CMS=1 /venv/dabai/bin/python ks_plain.py --ks $W/guard.mp4 --ks-lm $W/raw.mp.npz --master $M --box $B --lm $LF --audio $MIX --shift 0 \
  --lfm 6 --upper-src 0.12:0.32 --keep-nose 0.04 --align 2 --align-ecc $W/ecc.npz --align-scale 0 $OA --out $OUT </dev/null 2>&1 | grep -E "WARN|Trace|Error|plain paste ->"
./k_gate.sh $OUT $DUB $OUT.sync.json </dev/null
