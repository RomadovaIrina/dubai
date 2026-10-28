#!/bin/bash
# r7 F: labelled multi-way WATCH of the audio v3 variants rendered on the m8 face stack, one 2x2 grid per video, into the outbox:
#   <id>_r7_audio_v3_variants_WATCH.mp4 (v2 | v3a / v3b | v3c; four audio tracks, one per variant, titled; the first = v2)
#   bash scripts/r7/r7_variants_watch.sh "m8b:v2 m8v3a:v3a m8v3b:v3b m8v3c:v3c" 05 04
set -u
B=/workspace/dub/artifacts/r4m; OB=/workspace/dub/outbox; F=/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf; VARS=$1; shift
for id in ${*:-05 04}; do
  IN=""; FC=""; MAPS=""; META=""; k=0
  for v in $VARS; do
    tag=${v%%:*}; lab=${v##*:}; f=$B/$tag/${id}_$tag.mp4; [ -f $f ] || { echo "missing $f"; continue; }
    IN="$IN -i $f"; FC="$FC[$k:v]scale=1280:-2,drawtext=fontfile=$F:text='${lab} (${tag})':x=12:y=12:fontsize=40:fontcolor=yellow:box=1:boxcolor=black@0.6[v$k];"
    MAPS="$MAPS -map $k:a:0"; META="$META -metadata:s:a:$k title=$lab"; k=$((k+1))
  done
  case $k in
    4) FC="$FC[v0][v1]hstack[t];[v2][v3]hstack[b];[t][b]vstack,scale=1920:-2[v]";;
    3) FC="$FC[v0][v1][v2]hstack=inputs=3,scale=2560:-2[v]";;
    2) FC="$FC[v0][v1]hstack[v]";;
    *) echo "need 2-4 variants for $id"; continue;;
  esac
  ffmpeg -loglevel error -y $IN -filter_complex "$FC" -map "[v]" $MAPS $META -c:v libx264 -crf 23 -preset veryfast -pix_fmt yuv420p -c:a aac -b:a 160k $OB/${id}_r7_audio_v3_variants_WATCH.mp4 && echo "== variants WATCH $id ($k-way) $(date +%T)"
done
