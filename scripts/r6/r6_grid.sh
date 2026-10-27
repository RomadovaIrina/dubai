#!/bin/bash
# r6 G: 3x2 labelled grid of window clips (final-mix audio of the first arm), normal + half speed -> outbox/gen_<id>_<t0>s_ARMS[_half].mp4
#   bash scripts/r6/r6_grid.sh <id> <t0> ["arm:label" x6]   (arm SRC = the m3 master window = the source mouth)
id=$1; t0=$2; shift 2; O=/workspace/dub/artifacts/r6g/win; OB=/workspace/dub/outbox; F=/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf
A=("${@:-G0_base:m3 base g2.0 bridge}"); [ $# -eq 0 ] && A=("SRC:source (Russian mouth)" "G0_base:LS1.6 g2.0 (m3)" "G1_g25:LS1.6 g2.5" "G3_g25ds10:g2.5 + guidance smooth" "G9_s12nobr:g2.0 12 steps no bridge" "MT_musetalk:MuseTalk 1.5")
in=""; fc=""; k=0
for e in "${A[@]}"; do arm=${e%%:*}; lab=${e#*:}
  if [ $arm = SRC ]; then f=$O/master_${id}_${t0}.mp4; else f=$O/${id}_${arm}_${t0}.mp4; fi
  in="$in -i $f"; fc="$fc[$k:v]scale=464:-2,drawtext=fontfile=$F:text='$lab':x=6:y=6:fontsize=20:fontcolor=yellow:box=1:boxcolor=black@0.6[v$k];"; k=$((k+1)); done
fc="$fc[v0][v1][v2]hstack=3[r0];[v3][v4][v5]hstack=3[r1];[r0][r1]vstack=2[g]"
aud=1; ffmpeg -loglevel error -y $in -filter_complex "$fc" -map "[g]" -map $aud:a:0 -c:v libx264 -crf 20 -pix_fmt yuv420p -c:a aac -b:a 160k -shortest $OB/gen_${id}_${t0}s_ARMS.mp4
ffmpeg -loglevel error -y -i $OB/gen_${id}_${t0}s_ARMS.mp4 -filter_complex "[0:v]setpts=2*PTS[v];[0:a]atempo=0.5[a]" -map "[v]" -map "[a]" -c:v libx264 -crf 20 -pix_fmt yuv420p $OB/gen_${id}_${t0}s_ARMS_half.mp4
ls -la $OB/gen_${id}_${t0}s_ARMS*.mp4
