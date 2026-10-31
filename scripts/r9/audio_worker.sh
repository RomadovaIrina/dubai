#!/bin/bash
# r9 speed R: the resident audio worker (scripts/r9/audio_worker.py)
#   bash scripts/r9/audio_worker.sh start <spool dir>    one worker per GPU; then R9_WORKER_SPOOL=<spool dir> for e2e.sh / r8_audio.sh
#   bash scripts/r9/audio_worker.sh stop <spool dir>
set -u
cd "$(dirname "$0")/../.."; R=$(pwd); D=${2:?spool dir}
case "$1" in
  start)
    . scripts/env.sh >/dev/null 2>&1; export DUB_ROOT=$R PYTHONPATH=$R/third_party/llama_cuda${PYTHONPATH:+:$PYTHONPATH} PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True   # as r4_audio.sh starts the pipeline process
    mkdir -p $D; rm -f $D/stop $D/alive
    nohup python scripts/r9/audio_worker.py --spool $D > $D/worker.log 2>&1 < /dev/null & echo $! > $D/worker.pid
    for i in $(seq 100); do [ -f $D/alive ] && { echo "audio worker up (pid $(cat $D/worker.pid), spool $D)"; exit 0; }; sleep 0.2; done; echo "audio worker did not start (see $D/worker.log)"; exit 1;;
  stop) touch $D/stop; p=$(cat $D/worker.pid 2>/dev/null); [ -n "$p" ] && { for i in $(seq 150); do kill -0 $p 2>/dev/null || break; sleep 0.2; done; }; echo "audio worker stopped";;
esac
