#!/bin/bash
# r13 LS: q2 sweep = the colour-managed layers (r9g_<id>cmsf) + a LUMA-only, capped lip sharpen; per arm: Lab drift + sharpness, strong flashes
# (r3_flash, as the gate), quick SyncNet.   bash scripts/r12/q2.sh "<ids>" "tag=fps_composite flags" ...
cd /workspace/dub/dubai; B=/workspace/dub/artifacts/r4m; ids=$1; shift
for id in $ids; do for arm in "$@"; do t=${arm%%=*}; x=${arm#*=}; (
  GT=r9g_${id}cmsf FCX="$x" bash scripts/r12/comp_k5.sh $id $t FC_CMS=1 LS_LAYER_RGB=1 LS_CMS_READ=1 > /dev/null
  . scripts/env.sh >/dev/null; SRC=$(ls /workspace/dub/r10/runs/$id/work/in/$id.* | head -1); V=$B/r9f_$id$t/${id}_r9f_$id$t.mp4
  python scripts/r12/lab_drift.py --video $V --src $SRC --src-lm $B/r9g_$id/timing/${id}_source_lm_F.npz --label "$id $t"
  python scripts/quality/r3_flash.py --video $V --src $SRC --lm $B/r9g_$id/timing/${id}_source_lm_F.npz --out /workspace/dub/r12/flash_$id$t.json > /dev/null 2>&1
  python3 -c "import json;d=json.load(open('/workspace/dub/r12/flash_$id$t.json'));print('== flash $id $t strong', sum(1 for e in d['flash_list'] if (e.get('mc_ratio') or 0)>=2 and (e.get('mc_vs_src') or 0)>=2))"
  flock /workspace/dub/gpu.lock python scripts/r11/sync_speech.py --video $V --audio /workspace/dub/artifacts/r4a/r9_${id}_htd/$id/dubbed_16k.wav --label "$id $t" ) &
done; done; wait; echo Q2_DONE
