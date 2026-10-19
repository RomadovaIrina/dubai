#!/bin/bash
# r3 B: outbox finals with C's audio-r3b mix (same speech as r3, -16 LUFS): video of comp/<id>_c3_R3.mp4 + audio-r3b final_mix_aac.m4a, stream copy.
R=/workspace/dub/artifacts/r3-face; O=/workspace/dub/outbox
for id in "$@"; do
  A=/workspace/dub/handoff/in/C/audio-r3b/$id/final_mix_aac.m4a
  until [ -f $A ] && [ -f /workspace/dub/handoff/in/C/audio-r3b/READY ] && grep -q "^$id c3 done" $R/comp/c3_after.log; do sleep 30; done
  ffmpeg -loglevel error -y -i $R/comp/${id}_c3_R3.mp4 -i $A -map 0:v:0 -map 1:a:0 -c copy $R/comp/${id}_c3b_R3.mp4 && cp $R/comp/${id}_c3b_R3.mp4 $O/${id}_full_R3face_audio-C-r3b.mp4
  echo "$id r3b muxed $(date +%H:%M)"
done
