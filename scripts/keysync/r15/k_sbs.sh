#!/bin/bash
# r15 K: full-length side-by-side prodKS | fast2 of the cfhf finals (labels, the fast2 file audio = the q2 dub mix), <= 25 MB
F=/workspace/dub/outbox/r15_k/full; FT=$(fc-list 2>/dev/null | grep -i "DejaVuSans-Bold.ttf" | head -1 | cut -d: -f1); FF=""; [ -n "$FT" ] && FF="fontfile=$FT:"
DT="x=12:y=12:fontsize=30:fontcolor=white:box=1:boxcolor=black@0.6"
for v in m18 m03 old03; do L=$F/${v}_keysync_prodKS.mp4; R=$F/${v}_keysync_fast2.mp4; [ -s $L ] && [ -s $R ] || { echo "missing $v"; continue; }
  D=$(ffprobe -v error -show_entries format=duration -of csv=p=0 $R); BR=$(python3 -c "print(int(min(8000, 25e6 * 8 / $D * 0.92 / 1000 - 160)))")
  ffmpeg -v error -y -i $L -i $R -filter_complex "[0:v]scale=-2:960,setsar=1,drawtext=${FF}text=prod KeySync:$DT[l];[1:v]scale=-2:960,setsar=1,drawtext=${FF}text=fast KeySync 2-step:$DT[r];[l][r]hstack=inputs=2[v]" \
    -map "[v]" -map 1:a -c:v libx264 -preset slow -crf 20 -maxrate ${BR}k -bufsize $((2 * BR))k -pix_fmt yuv420p -movflags +faststart -c:a copy $F/${v}_prodKS_vs_fast2_full.mp4
  ls -la $F/${v}_prodKS_vs_fast2_full.mp4
done; echo K_SBS_DONE
