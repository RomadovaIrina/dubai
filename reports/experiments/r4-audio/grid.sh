#!/bin/bash
# 3x2 grid of 6 arms on one window (labelled, final-mix audio of the base arm), normal + half speed -> outbox/sweep_<id>_<t0>s_ARMS[_half].mp4
#   bash grid.sh <id> <t0>
id=$1; t0=$2; O=/workspace/dub/artifacts/r4sweep/win; OB=/workspace/dub/outbox; F=/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf
ARMS=(A0_base A5_seed7 A1_g15 A2_g25 A3_s30 A4_nobr); LAB=("base g2.0 s20 bridge seed1247" "seed 7" "guidance 1.5" "guidance 2.5" "30 steps" "bridge off")
in=""; fc=""; for k in 0 1 2 3 4 5; do in="$in -i $O/${id}_${ARMS[$k]}_${t0}.mp4"
  fc="$fc[$k:v]scale=640:-2,drawtext=fontfile=$F:text='${LAB[$k]}':x=8:y=8:fontsize=22:fontcolor=yellow:box=1:boxcolor=black@0.6[v$k];"; done
fc="$fc[v0][v1][v2]hstack=3[r0];[v3][v4][v5]hstack=3[r1];[r0][r1]vstack=2[g]"
ffmpeg -loglevel error -y $in -filter_complex "$fc" -map "[g]" -map 0:a:0 -c:v libx264 -crf 18 -pix_fmt yuv420p -c:a aac -b:a 160k $OB/sweep_${id}_${t0}s_ARMS.mp4
ffmpeg -loglevel error -y -i $OB/sweep_${id}_${t0}s_ARMS.mp4 -filter_complex "[0:v]setpts=2*PTS[v];[0:a]atempo=0.5[a]" -map "[v]" -map "[a]" -c:v libx264 -crf 18 -pix_fmt yuv420p $OB/sweep_${id}_${t0}s_ARMS_half.mp4
ls -la $OB/sweep_${id}_${t0}s_ARMS*.mp4
