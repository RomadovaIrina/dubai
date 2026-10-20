#!/bin/bash
# Round 4 V queue 3: the remaining one-layer-off ablations (03 / 05 / 01 in parallel, 02 has no lip-sync frames) and a 3-worker scoring pool
# (v4-repro, v4 reference, arms; arms reuse v4-repro's WER: same audio).
set -u
DUB_ROOT=/workspace/dub/dubai; . $DUB_ROOT/scripts/env.sh >/dev/null; cd $DUB_ROOT
B=/workspace/dub/artifacts/r4v; ARMS="no_retime no_sanitize no_sharpen no_colour no_skin no_faceclip no_seamlf no_lipguard"
for id in 03 05 01; do FIX30_THREADS=10 bash scripts/r4/r4_ablate.sh $id > $B/abl/q3_$id.out 2>&1 & done
{ for id in 03 05 01 02; do echo "v4repro $B/v4repro $id -"; echo "v4ref $B/ref $id $B/v4repro:v4repro"; done
  for id in 04 03 05 01; do for arm in $ARMS; do echo "$arm $B/abl $id ASR"; done; done; } > $B/q3_jobs.txt
score() {   # label dir id extra
  until [ -f $2/$3_$1.mp4 ] && { [ "$2" != "$B/abl" ] || [ $3 = 04 ] || grep -q "== $3 $1" $B/abl/q3_$3.out 2>/dev/null; }; do sleep 20; done
  if [ "$4" = ASR ]; then python scripts/r4/r4_gate.py score --label $1 --dir $2 --ids $3 --asr-from $B/v4repro:v4repro
  elif [ "$4" = - ]; then python scripts/r4/r4_gate.py score --label $1 --dir $2 --ids $3
  else python scripts/r4/r4_gate.py score --label $1 --dir $2 --ids $3 --manifest-from $4; fi
}
export -f score; export B
xargs -P 3 -L 1 bash -c 'score "$@"' _ < $B/q3_jobs.txt
wait; echo Q3_DONE
