#!/bin/bash
# wrap-up: stop the queue LOOPS (no new arms); the arm in flight keeps running. Prints what is still running.
for s in zcfg3.sh zcfg2.sh zcfg.sh zaudit.sh zflash_follow.sh chain_z.sh; do
  for p in $(pgrep -f "scripts/r15/$s|bash $s"); do kill $p 2>/dev/null && echo "stopped queue $s ($p)"; done
done
sleep 1; date +%T
ps -eo pid,etimes,args | grep -E "arm\.sh|candidate_video_stage|fps_composite|r7_m7_comp|sync_speech|lip_flicker|r3_flash|explode|comp_variant" | grep -v grep | cut -c1-140
