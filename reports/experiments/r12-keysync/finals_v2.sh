#!/bin/bash
# r12 KS finals v2: the 3 finals re-rendered with the lag fix on (ks_tsmooth alpha 0.8, the ks_final.sh default) -> outbox/r12_ks/final_v2;
# after each: SyncNet (speech, offset 0, 25 fps) + the lag spike rate, a push of the numbers
cd /workspace/dub/keysync; M=/workspace/dub/inputs/mamakris; OUT=/workspace/dub/outbox/r12_ks/final_v2; R=/workspace/dub/r12ks/reports/experiments/r12-keysync
for s in "q18 $M/story-1774250817106-616905094.mp4 m18" "m03 $M/story-1774249919482-367803614.MOV m03" "qo3 /workspace/dub/inputs/test_videos_raw/03.mp4 old03"; do set -- $s
  bash ks_final.sh $1 $2 $3 || { echo "FINAL_FAIL $1"; continue; }
  O=final/$1; F=$OUT/$3_KeySync.mp4; A=/workspace/dub/artifacts/r4a/r9_$1_htd/$1; SH=$(python3 -c "import json; print(-int(json.load(open('$O/raw_sync.json'))['global']['av_offset']))")
  echo "FINAL_READY $3 $(ls -la $F | awk '{print $5}')"
  ( cd /workspace/dub/dubai && /venv/mediapipe/bin/python scripts/quality/mp_landmarks.py --video $F --out /workspace/dub/keysync/$O/final_v2.mediapipe.npz | tail -1 )
  FPS=$(ffprobe -v error -select_streams v:0 -show_entries stream=r_frame_rate -of csv=p=0 $2 | python3 -c "import sys; n, d = sys.stdin.read().strip().split('/'); print(float(n) / float(d))")
  /venv/dabai/bin/python ks_lag.py --lm $O/final_v2.mediapipe.npz --src-lm $O/src.mediapipe.npz --fps $FPS --shift $SH --out $OUT/$3_lag | cut -c1-120
  ( cd /workspace/dub/dubai; . scripts/env.sh >/dev/null; export DUB_ROOT=/workspace/dub/dubai DUB_ALLOW_NC=1
    for v in /workspace/dub/outbox/r12_ks/final/$3_KeySync.mp4 $F; do n=v2_$1_$(echo $v | md5sum | cut -c1-6)
      ffmpeg -v error -y -i $v -vf fps=25 -an -c:v libx264 -crf 12 /tmp/s25_$n.mp4
      flock /workspace/dub/gpu.lock python scripts/quality/sync_perframe.py --video /tmp/s25_$n.mp4 --audio $A/dubbed_16k.wav --out /tmp/s_$n.json > /tmp/s_$n.log 2>&1
      python3 -c "import json,statistics as st; d=json.load(open('/tmp/s_$n.json')); sp=[p['conf_at0'] for p in d['per_second'] if p.get('conf_at0') is not None and (p.get('dub_act') or 0)>=0.5]; print('SYNC $3', '$v', d['global'], round(st.mean(sp),3))"
    done ) | tee $OUT/$3_sync.txt
  mkdir -p $R/finals_v2; cp ks_final.sh finals_v2.sh ks_tsmooth.py $R/; cp $OUT/$3_lag.json $OUT/$3_sync.txt $R/finals_v2/
  ( cd /workspace/dub/r12ks && git add -A reports/experiments/r12-keysync && git commit -q -m "r12 keysync: final v2 $3 with the lag fix on by default (ks_tsmooth alpha 0.8, KS_TSMOOTH=0 turns it off) + SyncNet / lag

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>" && git push -q origin experiment/r12-keysync 2>&1 | tail -1; git log -1 --format="PUSHED %h" )
done; echo FINALS_V2_DONE
