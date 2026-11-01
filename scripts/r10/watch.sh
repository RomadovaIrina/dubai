#!/bin/bash
# r10 FACE WATCH: full-video side by side, LEFT = r9 + fix (what management saw), RIGHT = the r10 face, labelled, the demucs mix, <= 25 MB.
#   bash scripts/r10/watch.sh <id> <right video> "<right label>"   -> /workspace/dub/outbox/r10_face/<id>_WATCH_r9fix_vs_r10face.mp4
set -u
id=$1; RV=$2; RL=${3:-r10 face (new)}; B=/workspace/dub/artifacts/r4m; A=/workspace/dub/artifacts/r4a; O=/workspace/dub/outbox/r10_face; mkdir -p $O
LV=$B/r9f_$id/${id}_r9f_$id.mp4; AUD=$A/r9_${id}_htd/$id/final_mix_aac.m4a; [ -f $AUD ] || AUD=$A/r9_$id/$id/final_mix_aac.m4a
OUT=$O/${id}_WATCH_r9fix_vs_r10face.mp4
dur=$(ffprobe -v error -show_entries format=duration -of csv=p=0 $LV); W=$(ffprobe -v error -select_streams v:0 -show_entries stream=width -of csv=p=0 $LV); H=$(ffprobe -v error -select_streams v:0 -show_entries stream=height -of csv=p=0 $LV)
# each panel <= 720 px high (portrait phones: 405x720 per panel); total bitrate for 24 MB incl. 128 kb/s audio
ph=$(( H > 960 ? 960 : H )); ph=$(( ph / 2 * 2 )); pw=$(( (W * ph / H) / 2 * 2 ))
vb=$(python3 -c "print(max(400, int(24 * 8 * 1000 / $dur - 140)))")
F=/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf; [ -f $F ] || F=$(fc-match -f '%{file}' sans 2>/dev/null)
lab() { echo "drawtext=fontfile=$F:text='$1':x=10:y=10:fontsize=$(( ph / 28 )):fontcolor=white:box=1:boxcolor=black@0.6:boxborderw=6"; }
ffmpeg -loglevel error -y -i $LV -i $RV -i $AUD -filter_complex \
  "[0:v]scale=$pw:$ph:flags=lanczos,$(lab 'r9 + fix (before)')[l];[1:v]scale=$pw:$ph:flags=lanczos,$(lab "$RL")[r];[l][r]hstack=inputs=2[v]" \
  -map "[v]" -map 2:a:0 -c:v libx264 -preset slow -b:v ${vb}k -maxrate $(( vb * 3 / 2 ))k -bufsize $(( vb * 2 ))k -pix_fmt yuv420p -c:a aac -b:a 128k -shortest -movflags +faststart $OUT
echo "== watch $OUT $(du -m $OUT | cut -f1) MB, ${vb} kb/s, ${pw}x${ph} per panel"
