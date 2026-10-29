#!/bin/bash
# r8 AB WATCH: <id>_<ver>_WATCH_AB_prev-then-new.mp4 = the source video with the PREV final mix, a 1-s black "B" card, then the same video
# with the NEW mix (NEW starts at PREV duration + 1 s), + <id>_<ver>_listen.md (coverage / pacing spans PREV vs NEW with both timecodes).
#   bash scripts/r8/r8_watch.sh <ver> <prev mix dir> <new tag> <ids>     e.g. r8_watch.sh r8v1 handoff/in_r7/audio-r7-v3a1 r8v1a 02 05
set -u
cd /workspace/dub/dubai; . scripts/env.sh >/dev/null
VER=$1; PREV=$2; NEW=$3; shift 3; PVER=${PVER:-v3a1}; A=/workspace/dub/artifacts/r4a; OUT=/workspace/dub/outbox; mkdir -p $OUT /workspace/dub/r8a/cov
for id in "$@"; do
  src=$(ls /workspace/dub/inputs/test_videos_raw/$id.* | head -1); t=$(mktemp -d)
  ffmpeg -loglevel error -y -i $src -i $PREV/$id/final_mix_aac.m4a -map 0:v:0 -map 1:a:0 -vf "scale=-2:720,fps=25,drawtext=text='A  $PVER':x=20:y=20:fontsize=36:fontcolor=white:box=1:boxcolor=black@0.6" \
    -c:v libx264 -crf 20 -preset veryfast -c:a aac -b:a 192k -ar 48000 -ac 2 -shortest $t/a.mp4
  W=$(ffprobe -v error -select_streams v:0 -show_entries stream=width,height -of csv=p=0 $t/a.mp4)
  ffmpeg -loglevel error -y -f lavfi -i "color=black:s=${W/,/x}:r=25:d=1" -f lavfi -i "anullsrc=r=48000:cl=stereo" -t 1 \
    -vf "drawtext=text='B  $VER':x=(w-tw)/2:y=(h-th)/2:fontsize=64:fontcolor=white" -c:v libx264 -crf 20 -preset veryfast -c:a aac -b:a 192k -shortest $t/card.mp4
  ffmpeg -loglevel error -y -i $src -i $A/$NEW/$id/final_mix_aac.m4a -map 0:v:0 -map 1:a:0 -vf "scale=-2:720,fps=25,drawtext=text='B  $VER':x=20:y=20:fontsize=36:fontcolor=white:box=1:boxcolor=black@0.6" \
    -c:v libx264 -crf 20 -preset veryfast -c:a aac -b:a 192k -ar 48000 -ac 2 -shortest $t/b.mp4
  printf "file '%s'\n" $t/a.mp4 $t/card.mp4 $t/b.mp4 > $t/l.txt
  ffmpeg -loglevel error -y -f concat -safe 0 -i $t/l.txt -c copy $OUT/${id}_audio_${VER}_WATCH_AB_prev-then-new.mp4
  ffmpeg -loglevel error -y -i $src -i $A/$NEW/$id/final_mix_aac.m4a -map 0:v:0 -map 1:a:0 -c copy -shortest $OUT/${id}_audio_${VER}_over-original.mp4
  off=$(python3 -c "import subprocess;d=float(subprocess.check_output(['ffprobe','-v','error','-show_entries','format=duration','-of','csv=p=0','$t/a.mp4']));print(round(d+1,2))")
  python scripts/r8/pacing.py --run $PREV/$id --id $id --label $PVER --face $A/sep/$id/source_mediapipe.npz --out /workspace/dub/r8a/pacing/prev_$id > /dev/null 2>&1
  python scripts/r8/coverage.py --run $PREV/$id --id $id --label prev --face $A/sep/$id/source_mediapipe.npz --out /workspace/dub/r8a/cov/prev_$id > /dev/null 2>&1
  python scripts/r8/listen.py --id $id --ver $VER --prev /workspace/dub/r8a/cov/prev_$id.json --new $A/$NEW/cov/cov_$id.json \
    --pace-prev /workspace/dub/r8a/pacing/prev_$id.json --pver $PVER --pace-new $A/$NEW/cov/pace_$id.json --offset $off > $OUT/${id}_audio_${VER}_listen.md
  rm -rf $t; echo "WATCH $id -> $OUT/${id}_audio_${VER}_WATCH_AB_prev-then-new.mp4 (B starts at $off s)"
done
