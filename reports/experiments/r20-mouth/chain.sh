#!/bin/bash
# chain.sh <job> <n>: cavity darkness along the chain of one job: source crop | raw | ts | snap | guard | seam | final (cropped by the box), the SAME landmarks (raw_lm) for the drawn ones
J=$1; n=$2; KW=/r18/data/jobs/$J/e2e/work/ks; P=$KW/post_f0; T=/r18/gen/chain_$J; mkdir -p $T; MP=/venv/mediapipe/bin/python
read x y s < <(python3 -c "import json; b=json.load(open(\"$KW/data/box_f0.json\")); print(b[\"x\"], b[\"y\"], b[\"side\"])")
ffmpeg -v error -y -i /r18/data/jobs/$J/$J.mp4 -an -vf "fps=25,crop=$s:$s:$x:$y,scale=512:512:flags=lanczos" -c:v libx264 -crf 10 -pix_fmt yuv420p $T/final.mp4
ffmpeg -v error -y -i $KW/master_25fps.mp4 -an -vf "crop=$s:$s:$x:$y,scale=512:512:flags=lanczos" -c:v libx264 -crf 10 -pix_fmt yuv420p $T/src.mp4
$MP /r18/x/cav2.py $n src=$T/src.mp4:$KW/crop_lm_f0.npz raw=$(ls $KW/out/f0/*.mp4 | head -1):$P/raw_lm.npz ts=$P/ts.mp4:$P/raw_lm.npz snap=$P/snap.mp4:$P/raw_lm.npz guard=$P/guard.mp4:$P/raw_lm.npz seam=$P/seam.mp4:$P/raw_lm.npz final=$T/final.mp4:$P/raw_lm.npz
