#!/bin/bash
# fc.sh <job> <ks dir in container> [ss offset s]: the final video cropped by the paste box to 512 (25 fps) -> /r18/gen/fc_<job>.mp4
J=$1; KW=$2; SS=${3:-0}
read x y s < <(python3 -c "import json; b=json.load(open(\"$KW/data/box_f0.json\")); print(b[\"x\"], b[\"y\"], b[\"side\"])")
ffmpeg -v error -y -ss $SS -i /r18/data/jobs/$J/$J.mp4 -an -vf "fps=25,crop=$s:$s:$x:$y,scale=512:512:flags=lanczos" -c:v libx264 -crf 10 -pix_fmt yuv420p /r18/gen/fc_$J.mp4
