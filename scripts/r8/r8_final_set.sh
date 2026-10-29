#!/bin/bash
# r8 F: the management set from one face render with several audio variants.
#   outbox/final_r8/<id>_DabAI_r8_<variant>.mp4 = the TAG render's video stream (copied, no re-encode) + the variant's final_mix_aac.m4a (copied),
#   faststart; <id>_DabAI_r8_<variant>_phone.mp4 (<= PHONE_MB, two-pass H.264 at the size-derived bitrate, same resolution, AAC 96 k).
#   TAG=m18 VARIANTS="demucs:/workspace/dub/handoff/in/audio-r8-v5 spleeter:/workspace/dub/handoff/in/audio-r8-v5-spleeter" bash scripts/r8/r8_final_set.sh [ids]
set -u
TAG=${TAG:-m18}; PHONE_MB=${PHONE_MB:-25}; B=/workspace/dub/artifacts/r4m/$TAG; OB=${OB:-/workspace/dub/outbox/final_r8}; mkdir -p $OB; T=$(mktemp -d)
VARIANTS=${VARIANTS:-"demucs:/workspace/dub/handoff/in/audio-r8-v5 spleeter:/workspace/dub/handoff/in/audio-r8-v5-spleeter"}
for id in ${*:-01 02 03 04 05}; do
  src=$B/${id}_$TAG.mp4; [ -f $src ] || { echo "no $src"; continue; }
  vdur=$(ffprobe -v error -select_streams v:0 -show_entries stream=duration -of csv=p=0 $src)
  for v in $VARIANTS; do
    name=${v%%:*}; mix=${v#*:}/$id/final_mix_aac.m4a; [ -f $mix ] || { echo "no $mix"; continue; }
    full=$OB/${id}_DabAI_r8_$name.mp4; phone=$OB/${id}_DabAI_r8_${name}_phone.mp4
    ffmpeg -loglevel error -y -i $src -i $mix -map 0:v:0 -map 1:a:0 -c copy -movflags +faststart $full
    adur=$(ffprobe -v error -select_streams a:0 -show_entries stream=duration -of csv=p=0 $full)
    dur=$(ffprobe -v error -show_entries format=duration -of csv=p=0 $full)
    vbr=$(python3 -c "d=$dur; print(int(max(200, (($PHONE_MB*0.95*8192) - 96*d - 0.015*$PHONE_MB*8192) / d)))")
    ffmpeg -loglevel error -y -i $full -map 0:v:0 -c:v libx264 -preset slow -b:v ${vbr}k -maxrate $((vbr*12/10))k -bufsize $((vbr*2))k -pass 1 -passlogfile $T/$id$name -an -f mp4 /dev/null
    ffmpeg -loglevel error -y -i $full -map 0:v:0 -map 0:a:0 -c:v libx264 -preset slow -b:v ${vbr}k -maxrate $((vbr*12/10))k -bufsize $((vbr*2))k -pass 2 -passlogfile $T/$id$name -pix_fmt yuv420p -c:a aac -b:a 96k -movflags +faststart $phone
    echo "== final $id $name: $(du -m $full | cut -f1) MB full, $(du -m $phone | cut -f1) MB phone (video ${vbr} kb/s; durations video $vdur / audio $adur s)"
  done
done
rm -rf $T; echo R8_FINAL_DONE $TAG
