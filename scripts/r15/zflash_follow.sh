#!/bin/bash
# r15 Z: run zflash.sh on each arm (alone + lk) of a video as its final appears (follows the running zcfg / zaudit queues)
#   bash zflash_follow.sh <id> <arm> ...
cd /workspace/dub/dubai; X=/workspace/dub/r15w/eval; id=$1; shift
for arm in "$@"; do
  for a in $arm ${arm}lk; do
    for i in $(seq 960); do grep -q "== sync" $X/${a}_$id.sync.txt 2>/dev/null && break; [ -f /workspace/dub/r15w/skip/${arm}_$id ] && break; sleep 30; done   # SyncNet runs on the FINISHED final
    [ -s $X/${a}_$id/${a}_$id.mp4 ] || continue
    grep -q "\"${a}_$id\"" $X/flash.jsonl 2>/dev/null || bash scripts/r15/zflash.sh $id $a $arm
  done
done
echo "ZFLASH_FOLLOW_DONE $id"
