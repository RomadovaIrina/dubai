#!/bin/bash
# r12 KS adaptive lag fix test: the 3 finals with KS_TSMOOTH=auto -> outbox/r12_ks/final_auto; SyncNet (speech, offset 0, 25 fps) + the lag spike rate
cd /workspace/dub/keysync; M=/workspace/dub/inputs/mamakris; OUT=/workspace/dub/outbox/r12_ks/final_auto
for s in "q18 $M/story-1774250817106-616905094.mp4 m18" "qo3 /workspace/dub/inputs/test_videos_raw/03.mp4 old03" "m03 $M/story-1774249919482-367803614.MOV m03"; do set -- $s
  KS_TSMOOTH=auto KS_OUTDIR=$OUT bash ks_final.sh $1 $2 $3 > final/$1/auto.log 2>&1 || { echo "FINAL_FAIL $1"; continue; }
  cp final/$1/layer_ts.mp4.alpha.json $OUT/$3_alpha.json; echo "ALPHA $3 $(cat $OUT/$3_alpha.json)"
  O=final/$1; F=$OUT/$3_KeySync.mp4; A=/workspace/dub/artifacts/r4a/r9_$1_htd/$1; SH=$(python3 -c "import json; print(-int(json.load(open('$O/raw_sync.json'))['global']['av_offset']))")
  ( cd /workspace/dub/dubai && /venv/mediapipe/bin/python scripts/quality/mp_landmarks.py --video $F --out /workspace/dub/keysync/$O/final_auto.mediapipe.npz > /dev/null 2>&1 )
  FPS=$(ffprobe -v error -select_streams v:0 -show_entries stream=r_frame_rate -of csv=p=0 $2 | python3 -c "import sys; n, d = sys.stdin.read().strip().split('/'); print(float(n) / float(d))")
  /venv/dabai/bin/python ks_lag.py --lm $O/final_auto.mediapipe.npz --src-lm $O/src.mediapipe.npz --fps $FPS --shift $SH --out $OUT/$3_lag | cut -c1-110
  ( cd /workspace/dub/dubai; . scripts/env.sh >/dev/null; export DUB_ROOT=/workspace/dub/dubai DUB_ALLOW_NC=1; n=auto_$1
    ffmpeg -v error -y -i $F -vf fps=25 -an -c:v libx264 -crf 12 /tmp/s25_$n.mp4
    flock /workspace/dub/gpu.lock python scripts/quality/sync_perframe.py --video /tmp/s25_$n.mp4 --audio $A/dubbed_16k.wav --out /tmp/s_$n.json > /tmp/s_$n.log 2>&1
    python3 -c "import json,statistics as st; d=json.load(open('/tmp/s_$n.json')); sp=[p['conf_at0'] for p in d['per_second'] if p.get('conf_at0') is not None and (p.get('dub_act') or 0)>=0.5]; print('SYNC $3 auto', d['global'], round(st.mean(sp),3))" ) | tee $OUT/$3_sync.txt
done; echo FINALS_AUTO_DONE
