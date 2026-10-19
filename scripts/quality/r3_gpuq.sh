#!/bin/bash
# Round 3 (track A) GPU queue: runs the lines of $Q (one shell command each) in order, one at a time; append lines any time.
# Done lines are moved to $Q.done with their exit code. Waits for any running LatentSync job of ours before starting the next.
Q=${1:-/workspace/dub/artifacts/r3/gpu.queue}; touch $Q; cd /workspace/dub/dubai
while true; do
  line=$(head -1 $Q)
  if [ -z "$line" ]; then sleep 15; continue; fi
  sed -i 1d $Q
  [ "${2:-gpu}" = gpu ] && while pgrep -f "candidate_video_stage.py|ls_window_ab.py" >/dev/null; do sleep 10; done
  echo "[$(date +%H:%M:%S)] START $line" >> $Q.done
  bash -c "$line" >> $Q.log 2>&1; rc=$?
  echo "[$(date +%H:%M:%S)] END rc=$rc $line" >> $Q.done
done
