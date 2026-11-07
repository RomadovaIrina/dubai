#!/bin/bash
# r16 K2: k2_var.sh <v> <mode: plain|lift|bank|synth> [ks.mp4]  -> /workspace/dub/r16k2/<v>/<mode>/{ks.mp4,final.mp4,cavity.json}
v=$1; m=$2; P=/workspace/dub/r16in/HANDOFF_K/paste_inputs/$v; R=/workspace/dub/r16k2; O=$R/$v/$m; mkdir -p $O
KS=${3:-$P/keysync_fast2_raw512.mp4}; KL=${KSLM:-$P/keysync_raw512.mediapipe.npz}; PY=/venv/mediapipe/bin/python; K=/workspace/dub/keysync
SH=$(python3 -c "import json; print(-int(json.load(open('$P/raw_sync.json'))['global']['av_offset']))")
OTH=""; [ -f $P/faces2.npz ] && OTH="--others $P/faces2.npz"
case $m in plain|plain_*|b4|b8|s2own|g4|g8|rt|pa|pb) cp -f $KS $O/ks.mp4;; *) $PY $R/k2_mouth.py --mode ${MODE:-$m} --ks $KS --ks-lm $KL --master $R/src/${v}_master_25fps.mp4 --box $P/box.json --lm $P/source_lm_F.npz $OTH --out $O/ks.mp4 --json $O/mouth.json $K2ARGS > $O/mouth.log 2>&1;; esac
cd $K; KS_CMS=1 $PY ks_plain.py --ks $O/ks.mp4 --master $R/src/${v}_master_25fps.mp4 --box $P/box.json --lm $P/source_lm_F.npz $OTH --shift $SH --audio $P/final_mix_aac.m4a --out $O/final.mp4 --ks-lm $KL $PLAINARGS > $O/plain.log 2>&1
$PY $K/k_cavity.py --pairs $m=$O/ks.mp4:$KL > $O/cavity.json 2>$O/cavity.err
echo "K2_VAR $v $m $(python3 -c "import json;d=json.load(open('$O/cavity.json'));x=list(d.values())[0];print(x['black_hole_share'],x['open_frames'],x['cavity_L_mean'],x['teeth_bright_share'])")"
