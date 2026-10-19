#!/bin/bash
# Round 3 track B: GPU job queue (one LatentSync-class job at a time: one needs ~21.5 GB of the 32 GB, two OOM). Jobs = lines of
# $Q/queue.txt (shell commands, run from the repo root); the first line is popped and run, its log in $Q/logs/. Idle -> polls every 20 s.
#   nohup scripts/quality/r3_gpuq.sh &      echo "<command>" >> /workspace/dub/artifacts/r3-face/gpuq/queue.txt
Q=/workspace/dub/artifacts/r3-face/gpuq; mkdir -p $Q/logs; touch $Q/queue.txt; cd /workspace/dub/dubai
while true; do
  [ -f $Q/STOP ] && exit 0
  job=$(head -1 $Q/queue.txt)
  if [ -z "$job" ]; then sleep 20; continue; fi
  sed -i '1d' $Q/queue.txt; n=$(date +%m%d_%H%M%S)
  echo "$n START $job" >> $Q/done.txt
  bash -c "$job" > $Q/logs/$n.log 2>&1; echo "$(date +%m%d_%H%M%S) END rc=$? $job" >> $Q/done.txt
done
