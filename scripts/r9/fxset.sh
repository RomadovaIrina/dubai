#!/bin/bash
# r9fx set: fx composites (layer repair v2 + inner-mouth detail 0.4) on the shipped r9 generator layers, then both mixes per video -> outbox/r9fx/
cd /workspace/dub/dubai; . scripts/env.sh >/dev/null; export DUB_ROOT=/workspace/dub/dubai DUB_ALLOW_NC=1 METRICS=off
A=/workspace/dub/artifacts/r4a; B=/workspace/dub/artifacts/r4m; D=/workspace/dub/r9/diag; O=/workspace/dub/outbox/r9fx; mkdir -p $O; FBA="$(cat reports/experiments/r8-face/fb7.args)"
for id in "$@"; do export DUB_INPUTS=/workspace/dub/r9/runs/$id/work/in
  [ -f $B/r9f_${id}fx/${id}_r9f_${id}fx.mp4 ] || SKIP_SCORE=1 TAG=r9f_${id}fx SRC_TAG=r9g_$id CFD=self CFD_LABEL=ia04 CFD_ARGS="--inner-alpha 0.4" SR=srt1 SR_EXTRA="--batch 8 --mask down2" FB=fb7 FB_ARGS="$FBA" REP=1 bash scripts/r7/r7_m7_comp.sh $id > $D/fx_$id.log 2>&1
  echo "== r9f_${id}fx $(grep -h layer_repair $D/fx_$id.log 2>/dev/null | cut -c1-200) $(date +%T)"
done; echo FXSET_DONE
