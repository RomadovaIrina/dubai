#!/bin/bash
cd /workspace/dub/dubai; . scripts/env.sh >/dev/null 2>&1; export DUB_ROOT=/workspace/dub/dubai DUB_ALLOW_NC=1
for v in m18 m03 old03; do ffmpeg -v error -y -i /workspace/dub/outbox/r15_k/full/${v}_keysync_${PNAME:-fast2plain}.mp4 -vf fps=25 -c:v libx264 -crf 12 -an /tmp/p25_$v.mp4
  flock /workspace/dub/gpu.lock python scripts/quality/sync_perframe.py --video /tmp/p25_$v.mp4 --audio /workspace/dub/dubaudio/$v/dubbed_16k.wav --out /tmp/sp_$v.json > /dev/null 2>&1
  python3 -c "import json,statistics as st; d=json.load(open(\"/tmp/sp_$v.json\")); sp=[p[\"conf_at0\"] for p in d[\"per_second\"] if p.get(\"conf_at0\") is not None and (p.get(\"dub_act\") or 0)>=0.5]; print(\"SYNC_${PNAME:-fast2plain} $v\", round(st.mean(sp),3), d[\"global\"])"; done
