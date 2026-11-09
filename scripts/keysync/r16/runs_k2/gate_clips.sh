#!/bin/bash
# 0.5x eye clips of the wide-open moments: gate off | g8 (current) | g3 (hi=3) | source
B=/workspace/dub/r16k2/gate; O=/workspace/dub/outbox/r16_k2/gate; mkdir -p $O; F=/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf
:
for spec in "m18 2 6 720:720:0:320" "m03 23 27 476:476:223:246"; do set -- $spec; v=$1; t0=$2; t1=$3; cr=$4; d=$((t1 - t0)); fc=""; ins=""; i=0
  for lv in "gate_off:$B/$v/off.mp4" "gate_hi8_now:$B/$v/g8.mp4" "gate_hi3:$B/$v/g3.mp4" "source:$B/$v/w_off/ks/master_25fps.mp4"; do l=${lv%%:*}; f=${lv#*:}; ins="$ins -ss $t0 -t $d -i $f"
    fc="$fc[$i:v]crop=$cr,scale=400:400,setpts=PTS-STARTPTS,drawtext=fontfile=$F:text=$l:x=6:y=6:fontsize=20:fontcolor=yellow:box=1:boxcolor=black@0.5,drawtext=fontfile=$F:text='%{pts\:hms\:$t0}':x=6:y=h-30:fontsize=18:fontcolor=white:box=1:boxcolor=black@0.5,setpts=2*PTS[v$i];"; i=$((i+1)); done
  ffmpeg -nostdin -v error -y $ins -ss $t0 -t $d -i $B/$v/ad/final_mix_aac.m4a -filter_complex "${fc}[v0][v1][v2][v3]hstack=inputs=4[v];[4:a]asetpts=PTS-STARTPTS,atempo=0.5[a]" -map "[v]" -map "[a]" \
    -c:v libx264 -crf 22 -preset medium -pix_fmt yuv420p -r 25 -c:a aac -b:a 96k -movflags +faststart $O/${v}_gateoff_hi8_hi3_source_${t0}-${t1}s_half_speed.mp4 && echo "CLIP $O/${v}_gateoff_hi8_hi3_source_${t0}-${t1}s_half_speed.mp4 $(stat -c %s $O/${v}_gateoff_hi8_hi3_source_${t0}-${t1}s_half_speed.mp4)"
done
