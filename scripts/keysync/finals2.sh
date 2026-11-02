#!/bin/bash
# r12 KS finals in the coordinator's order (m18, m03, old03); after each: the lag analysis + a push of the scripts / numbers
cd /workspace/dub/keysync; M=/workspace/dub/inputs/mamakris
until grep -q E2E3_DONE /workspace/dub/r12k/e2e3.log; do sleep 15; done
for s in "q18 $M/story-1774250817106-616905094.mp4 m18" "m03 $M/story-1774249919482-367803614.MOV m03" "qo3 /workspace/dub/inputs/test_videos_raw/03.mp4 old03"; do set -- $s
  bash ks_final.sh $1 $2 $3 || { echo "FINAL_FAIL $1"; continue; }
  O=final/$1; F=/workspace/dub/outbox/r12_ks/final/$3_KeySync.mp4; SH=$(python3 -c "import json; print(-int(json.load(open('$O/raw_sync.json'))['global']['av_offset']))")
  ( cd /workspace/dub/dubai && /venv/mediapipe/bin/python scripts/quality/mp_landmarks.py --video $F --out /workspace/dub/keysync/$O/final.mediapipe.npz | tail -1
    /venv/mediapipe/bin/python scripts/quality/mp_landmarks.py --video $2 --out /workspace/dub/keysync/$O/src.mediapipe.npz | tail -1 )
  FPS=$(ffprobe -v error -select_streams v:0 -show_entries stream=r_frame_rate -of csv=p=0 $2 | python3 -c "import sys; n, d = sys.stdin.read().strip().split('/'); print(float(n) / float(d))")
  /venv/dabai/bin/python ks_lag.py --lm $O/final.mediapipe.npz --src-lm $O/src.mediapipe.npz --fps $FPS --shift $SH --out /workspace/dub/outbox/r12_ks/final/$3_lag
  R=/workspace/dub/r12ks/reports/experiments/r12-keysync; mkdir -p $R/finals; cp ks_final.sh ks_lag.py finals2.sh ks_prep_src.sh ks_eval.sh $R/; cp /workspace/dub/outbox/r12_ks/final/$3_lag.json $R/finals/; cp $O/raw_sync.json $R/finals/$3_raw_sync.json; cp $O/guard.json $R/finals/$3_guard.json 2>/dev/null
  ( cd /workspace/dub/r12ks && git add -A reports/experiments/r12-keysync && git commit -q -m "r12 keysync: final $3 (10 steps, AV shift $SH, ks_fix guard 3 + chin tex, raw paste, demucs mix) + lag analysis

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>" && git push -q origin experiment/r12-keysync 2>&1 | tail -1; git log -1 --format="PUSHED %h" )
  echo "FINAL_READY $3 $(ls -la $F | awk '{print $5}')"
done; echo FINALS2_DONE
