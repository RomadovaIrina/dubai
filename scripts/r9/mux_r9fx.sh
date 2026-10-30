#!/bin/bash
# both mixes on the r9fx picture: <id>_DabAI_r9fx_{demucs,spleeter}.mp4 (m03 = the fx3 rebuild; a video without a face = the r9 picture)
A=/workspace/dub/artifacts/r4a; B=/workspace/dub/artifacts/r4m; O=/workspace/dub/outbox/r9fx; mkdir -p $O
for id in "$@"; do T=r9f_${id}fx; [ $id = m03 ] && T=r9f_m03fx3; V=$B/$T/${id}_$T.mp4; [ -f $V ] || V=/workspace/dub/r9/runs/$id/${id}_DabAI_r9_spleeter.mp4
  for v in demucs:$A/r9_${id}_htd/$id/final_mix_aac.m4a spleeter:$A/r9_$id/$id/final_mix_aac.m4a; do ffmpeg -loglevel error -y -i $V -i ${v#*:} -map 0:v:0 -map 1:a:0 -c copy -movflags +faststart $O/${id}_DabAI_r9fx_${v%%:*}.mp4; done
  echo "== $id picture $(basename $V)"; done
