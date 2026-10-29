#!/bin/bash
# r8-sep AB WATCH: <id>_audio_sep_WATCH_AB_${WATCH_NAME:-r8v4-then-sep}.mp4 = the source video with the r8v4 mix (A, Spleeter bed), a 1-s black "B" card, then the
# same video with the sep mix (B), + <id>_audio_sep_over-original.mp4 + <id>_audio_sep_listen.md (scripts/r8/listen_sep.py, B = A + offset).
#   bash scripts/r8/sep_watch.sh <new tag> <ids>
set -u
cd /workspace/dub/sep; . scripts/env.sh >/dev/null; export DUB_ROOT=/workspace/dub/sep CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=6
NEW=$1; shift; A=/workspace/dub/artifacts/r4a; PREV=$A/${WATCH_PREV:-r8v4c}; OUT=/workspace/dub/outbox/sep; mkdir -p $OUT
for id in "$@"; do
  src=$(ls /workspace/dub/inputs/test_videos_raw/$id.* | head -1); t=$(mktemp -d)
  ffmpeg -loglevel error -y -i $src -i $PREV/$id/final_mix_aac.m4a -map 0:v:0 -map 1:a:0 -vf "scale=-2:720,fps=25,drawtext=text='A  ${WATCH_PREV_LABEL:-r8v4} (Spleeter bed)':x=20:y=20:fontsize=36:fontcolor=white:box=1:boxcolor=black@0.6" \
    -c:v libx264 -crf 20 -preset veryfast -c:a aac -b:a 192k -ar 48000 -ac 2 -shortest $t/a.mp4
  W=$(ffprobe -v error -select_streams v:0 -show_entries stream=width,height -of csv=p=0 $t/a.mp4)
  ffmpeg -loglevel error -y -f lavfi -i "color=black:s=${W/,/x}:r=25:d=1" -f lavfi -i "anullsrc=r=48000:cl=stereo" -t 1 \
    -vf "drawtext=text='B  sep (htdemucs bed)':x=(w-tw)/2:y=(h-th)/2:fontsize=56:fontcolor=white" -c:v libx264 -crf 20 -preset veryfast -c:a aac -b:a 192k -shortest $t/card.mp4
  ffmpeg -loglevel error -y -i $src -i $A/$NEW/$id/final_mix_aac.m4a -map 0:v:0 -map 1:a:0 -vf "scale=-2:720,fps=25,drawtext=text='B  sep (htdemucs bed)':x=20:y=20:fontsize=36:fontcolor=white:box=1:boxcolor=black@0.6" \
    -c:v libx264 -crf 20 -preset veryfast -c:a aac -b:a 192k -ar 48000 -ac 2 -shortest $t/b.mp4
  printf "file '%s'\n" $t/a.mp4 $t/card.mp4 $t/b.mp4 > $t/l.txt
  ffmpeg -loglevel error -y -f concat -safe 0 -i $t/l.txt -c copy $OUT/${id}_audio_sep_WATCH_AB_${WATCH_NAME:-r8v4-then-sep}.mp4
  ffmpeg -loglevel error -y -i $src -i $A/$NEW/$id/final_mix_aac.m4a -map 0:v:0 -map 1:a:0 -c copy -shortest $OUT/${id}_audio_sep_over-original.mp4
  off=$(python3 -c "import subprocess;d=float(subprocess.check_output(['ffprobe','-v','error','-show_entries','format=duration','-of','csv=p=0','$t/a.mp4']));print(round(d+1,2))")
  python scripts/r8/listen_sep.py --id $id --prev ${WATCH_PREV:-r8v4c} --new $NEW --offset $off > $OUT/${id}_audio_sep_listen.md 2>/dev/null
  rm -rf $t; echo "WATCH $id -> $OUT/${id}_audio_sep_WATCH_AB_${WATCH_NAME:-r8v4-then-sep}.mp4 (B starts at $off s)"
done
