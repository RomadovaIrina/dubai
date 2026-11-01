#!/bin/bash
# r10 speed: the resident face-generator worker (scripts/r10/gen_worker.py)
#   bash scripts/r10/gen_worker.sh start <spool dir>    one per GPU; then R10_GEN_SPOOL=<spool dir> for e2e.sh (r4_video.sh sends the generator job to it)
#   bash scripts/r10/gen_worker.sh stop <spool dir>
# The worker's own environment fixes the compiled graphs: LS_VAE_COMPILE / LS_UNET_COMPILE as e2e.sh would set them.
set -u
cd "$(dirname "$0")/../.."; R=$(pwd); D=${2:?spool dir}
case "$1" in
  start)
    . scripts/env.sh >/dev/null 2>&1; export DUB_ROOT=$R PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True DUB_ALLOW_NC=1
    export LS_VAE_COMPILE=${LS_VAE_COMPILE-max-autotune-no-cudagraphs}
    mkdir -p $D/jobs $D/out; rm -f $D/stop $D/alive
    nohup python scripts/r10/gen_worker.py --spool $D > $D/worker.log 2>&1 < /dev/null & echo $! > $D/worker.pid
    for i in $(seq 600); do grep -q "gen_worker\] up" $D/worker.log 2>/dev/null && { echo "generator worker up (pid $(cat $D/worker.pid), spool $D)"; exit 0; }
      kill -0 $(cat $D/worker.pid) 2>/dev/null || break; sleep 0.5; done; echo "generator worker did not start (see $D/worker.log)"; exit 1;;
  stop) touch $D/stop; p=$(cat $D/worker.pid 2>/dev/null); [ -n "$p" ] && { for i in $(seq 150); do kill -0 $p 2>/dev/null || break; sleep 0.2; done; }; echo "generator worker stopped";;
esac
