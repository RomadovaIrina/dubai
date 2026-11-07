#!/bin/bash
# r16 L: (re)compute the CPU SyncNet + flow for every eval dir given, then re-aggregate.   l_resync.sh <eval dir> ...
R=/workspace/dub/r16l; L=$R/repo/scripts/keysync/r16/l
for O in "$@"; do for G in $O/*/*_*.mp4; do tag=$(basename $(dirname $G)); aud=$R/data/audios/$tag.wav
  [ -s $O/$tag.sync.json ] && grep -q cpu $O/$tag.sync.dev 2>/dev/null || { (cd /workspace/dub/dubai && SYNC_DEVICE=cpu /venv/dabai/bin/python $R/repo/scripts/quality/sync_perframe.py --video $G --audio $aud --out $O/$tag.sync.json > $O/$tag.sync.log 2>&1) && echo cpu > $O/$tag.sync.dev; }
  [ -s $O/$tag.flow.json ] || /venv/dabai/bin/python $L/l_flow.py --gen $G --gen-lm $O/$tag.mp.npz --src $R/data/videos/$tag.mp4 --src-lm $R/data/videos/$tag.mp.npz --out $O/$tag.flow.json > /dev/null 2>&1
done; python3 $L/l_agg.py $O; done
