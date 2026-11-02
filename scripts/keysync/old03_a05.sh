#!/bin/bash
# old03 final with the lag fix at alpha 0.5 (v2 at 0.8 cost SyncNet -0.29 on a video with no excess jitter)
cd /workspace/dub/keysync; OUT=/workspace/dub/outbox/r12_ks/final_v2_a05; A=/workspace/dub/artifacts/r4a/r9_qo3_htd/qo3
KS_TSMOOTH=0.5 KS_OUTDIR=$OUT bash ks_final.sh qo3 /workspace/dub/inputs/test_videos_raw/03.mp4 old03 || exit 1
F=$OUT/old03_KeySync.mp4; echo "FINAL_READY old03_a05 $(ls -la $F | awk '{print $5}')"
cd /workspace/dub/dubai; . scripts/env.sh >/dev/null; export DUB_ROOT=/workspace/dub/dubai DUB_ALLOW_NC=1
ffmpeg -v error -y -i $F -vf fps=25 -an -c:v libx264 -crf 12 /tmp/s25_o03a05.mp4
flock /workspace/dub/gpu.lock python scripts/quality/sync_perframe.py --video /tmp/s25_o03a05.mp4 --audio $A/dubbed_16k.wav --out /tmp/s_o03a05.json > /dev/null 2>&1
python3 -c "import json,statistics as st; d=json.load(open('/tmp/s_o03a05.json')); sp=[p['conf_at0'] for p in d['per_second'] if p.get('conf_at0') is not None and (p.get('dub_act') or 0)>=0.5]; print('SYNC old03_a05', d['global'], round(st.mean(sp),3))" | tee $OUT/old03_sync.txt
echo O03A05_DONE
