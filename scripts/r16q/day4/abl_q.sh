#!/bin/bash
# queue runner: lines "ARM|video|ENV|EXTRA" ; at most $1 runs at once (16 at once exhausted threads on r16-a: ffmpeg EAGAIN)
cd /workspace/dub/q_abl; P=${1:-6}; export ABL_JOBS=${ABL_JOBS:-24}
one(){ IFS='|' read -r arm v envs extra <<< "$1"; rm -rf work_$arm/jobs/r16d_$v out/$arm/$v.mp4; ABL_EXTRA="$extra" ./abl.sh $arm "$envs" $v; }
export -f one
grep -v '^#' queue.txt | grep . | xargs -d '\n' -P $P -I{} bash -c 'one "$@"' _ {}
echo QUEUE_DONE >> done.txt
