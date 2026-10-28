#!/bin/bash
# r7 F: the final deliverable set from a merged candidate: outbox/final_r7/<id>_DabAI_r7.mp4 (plain dubbed video, source resolution / fps,
# H.264 + AAC, faststart; the candidate's own encode remuxed) + <id>_DabAI_r7_phone.mp4 (<= PHONE_MB, two-pass H.264 at the size-derived
# bitrate, same resolution, AAC 96 k).   TAG=m11 PHONE_MB=25 bash scripts/r7/r7_final_set.sh [ids]
set -u
TAG=${TAG:-m11}; PHONE_MB=${PHONE_MB:-25}; B=/workspace/dub/artifacts/r4m/$TAG; OB=/workspace/dub/outbox/final_r7; mkdir -p $OB; T=$(mktemp -d)
for id in ${*:-01 02 03 04 05}; do
  src=$B/${id}_$TAG.mp4; [ -f $src ] || { echo "no $src"; continue; }
  ffmpeg -loglevel error -y -i $src -map 0:v:0 -map 0:a:0 -c copy -movflags +faststart $OB/${id}_DabAI_r7.mp4
  dur=$(ffprobe -v error -show_entries format=duration -of csv=p=0 $src); fps=$(ffprobe -v error -select_streams v:0 -show_entries stream=r_frame_rate -of csv=p=0 $src)
  # target: 95 % of the size cap minus the audio and ~1.5 % container overhead
  vbr=$(python3 -c "d=$dur; print(int(max(200, (($PHONE_MB*0.95*8192) - 96*d - 0.015*$PHONE_MB*8192) / d)))")
  ffmpeg -loglevel error -y -i $src -map 0:v:0 -map 0:a:0 -c:v libx264 -preset slow -b:v ${vbr}k -maxrate $((vbr*12/10))k -bufsize $((vbr*2))k -pass 1 -passlogfile $T/$id -an -f mp4 /dev/null
  ffmpeg -loglevel error -y -i $src -map 0:v:0 -map 0:a:0 -c:v libx264 -preset slow -b:v ${vbr}k -maxrate $((vbr*12/10))k -bufsize $((vbr*2))k -pass 2 -passlogfile $T/$id -pix_fmt yuv420p -c:a aac -b:a 96k -movflags +faststart $OB/${id}_DabAI_r7_phone.mp4
  echo "== final $id: $(du -m $OB/${id}_DabAI_r7.mp4 | cut -f1) MB full, $(du -m $OB/${id}_DabAI_r7_phone.mp4 | cut -f1) MB phone (video ${vbr} kb/s, ${dur%.*} s, $fps fps)"
done
rm -rf $T; echo R7_FINAL_DONE $TAG
