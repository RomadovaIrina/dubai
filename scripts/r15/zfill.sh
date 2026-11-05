#!/bin/bash
# wrap-up: fill the missing SyncNet cells (zfinal.sh, no panels), q2as old03 blinks, q2nr m18 post when its composite ends
cd /workspace/dub/dubai; . scripts/env.sh >/dev/null
for p in $(pgrep -f "zpost.sh old03"); do kill $p; done
bash scripts/r15/zfinal.sh m18 old03 > /dev/null 2>&1
bash scripts/r15/zlf.sh old03 q2as q2as
for i in $(seq 40); do pgrep -f "r9g_q2_m18_q2nr" >/dev/null || break; sleep 15; done
X=/workspace/dub/r15w/eval
for a in q2nr q2nrlk; do [ -s $X/${a}_m18/${a}_m18.mp4 ] || continue
  flock /workspace/dub/gpu.lock python scripts/r11/sync_speech.py --video $X/${a}_m18/${a}_m18.mp4 --audio /workspace/dub/artifacts/r4a/r9_q2_m18/q2_m18/dubbed_16k.wav --label ${a}_m18 > $X/${a}_m18.sync.txt 2>&1
  bash scripts/r15/zlf.sh m18 $a q2nr; done
echo ZFILL_DONE
