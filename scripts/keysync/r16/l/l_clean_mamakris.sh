#!/bin/bash
# r16 L: a CLEAN training list (brief r16b, audit #5): the mamakris stories that are NOT m03 / m18, with their OWN audio, as KeySync 512 crops
# (ks_prep.py: static face box from MediaPipe landmarks). Same speaker / room as m03 / m18 (as the r15 init), but no test clip.
#   l_clean_mamakris.sh [out dir = /workspace/dub/r16l/clean]  -> <out>/clean_list.txt, then: l_train_jobs.sh <out>/clean_list.txt jobs.txt
set -e
O=${1:-/workspace/dub/r16l/clean}; M=/workspace/dub/inputs/mamakris; K=/workspace/dub/keysync; Q=/workspace/dub/dubai/scripts/quality
mkdir -p $O/videos $O/audios $O/lm; : > $O/clean_list.txt
for f in $M/*; do b=$(basename $f); case $b in story-1774249919482-*|story-1774250817106-*) echo "skip test clip $b"; continue;; esac
  t=mk_$(echo $b | cut -d- -f2 | tail -c 7)
  ffmpeg -v error -y -i $f -an -vf fps=25 -c:v libx264 -crf 14 -pix_fmt yuv420p $O/lm/$t.25.mp4
  /venv/mediapipe/bin/python $Q/mp_landmarks.py --video $O/lm/$t.25.mp4 --out $O/lm/$t.npz > /dev/null 2>&1
  ffmpeg -v error -y -i $f -vn -ac 1 -ar 16000 $O/lm/$t.wav
  (cd $K && /venv/dabai/bin/python $K/ks_prep.py $O/lm/$t.25.mp4 $O/lm/$t.npz $O/lm/$t.wav $O $t) && echo "$t $O/videos/$t.mp4 $O/audios/$t.wav" >> $O/clean_list.txt
done
wc -l $O/clean_list.txt
