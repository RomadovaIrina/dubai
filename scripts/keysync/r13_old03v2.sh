#!/bin/bash
# r13 URGENT: old03 KeySync v2 = the finals path (10 steps, guard + chin tex, alpha 0.8 smoothing, raw paste, demucs mix, <= 25 MB) on the regenerated r13 dub;
# then the same with KS_TSMOOTH=0 (v1 on this dub) for the SyncNet / jerk comparison. The m03 E2E runs after the KeySync stage (VRAM).
cd /workspace/dub/keysync; OUT=/workspace/dub/outbox/r13_ks; mkdir -p $OUT; SRC=/workspace/dub/inputs/test_videos_raw/03.mp4; O=final/qo3; A=/workspace/dub/artifacts/r4a/r9_qo3_htd/qo3
meas() {  # <video> <tag>: SyncNet speech at0 (25 fps) + the lag spike rate
  ( cd /workspace/dub/dubai && /venv/mediapipe/bin/python scripts/quality/mp_landmarks.py --video $1 --out /workspace/dub/keysync/$O/$2.mediapipe.npz > /dev/null 2>&1 )
  SH=$(python3 -c "import json; print(-int(json.load(open('$O/raw_sync.json'))['global']['av_offset']))")
  /venv/dabai/bin/python ks_lag.py --lm $O/$2.mediapipe.npz --src-lm $O/src.mediapipe.npz --fps 25 --shift $SH --out $OUT/$2_lag | python3 -c "import json,sys; d=json.loads(sys.stdin.read()); print('LAG $2', d['spike_rate'], 'src', d['source_rate'])"
  ( cd /workspace/dub/dubai; . scripts/env.sh >/dev/null; export DUB_ROOT=/workspace/dub/dubai DUB_ALLOW_NC=1
    ffmpeg -v error -y -i $1 -vf fps=25 -an -c:v libx264 -crf 12 /tmp/s25_$2.mp4
    flock /workspace/dub/gpu.lock python scripts/quality/sync_perframe.py --video /tmp/s25_$2.mp4 --audio $A/dubbed_16k.wav --out /tmp/s_$2.json > /tmp/s_$2.log 2>&1
    python3 -c "import json,statistics as st; d=json.load(open('/tmp/s_$2.json')); sp=[p['conf_at0'] for p in d['per_second'] if p.get('conf_at0') is not None and (p.get('dub_act') or 0)>=0.5]; print('SYNC $2', d['global'], round(st.mean(sp),3))" )
}
KS_OUTDIR=$OUT bash ks_final.sh qo3 $SRC old03_v2 > $O/final_v2.log 2>&1 || { echo FINAL_FAIL; tail -5 $O/final_v2.log; }
mv $OUT/old03_v2_KeySync.mp4 $OUT/old03_KeySync_v2.mp4; echo "FINAL_READY $(ls -la $OUT/old03_KeySync_v2.mp4 | awk '{print $5}')"
( bash r13_e2e_m03.sh >> r13_e2e.log 2>&1 & )
FPS=25; meas $OUT/old03_KeySync_v2.mp4 old03_v2
KS_TSMOOTH=0 KS_OUTDIR=$OUT/v1check bash ks_final.sh qo3 $SRC old03_v1 > $O/final_v1check.log 2>&1
meas $OUT/v1check/old03_v1_KeySync.mp4 old03_v1
echo OLD03V2_DONE
