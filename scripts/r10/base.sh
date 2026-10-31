#!/bin/bash
# r10 FACE: r9 + fix baselines (E2E_FX=1, production only) + the fx + stab composite of the same layers, per id.
#   bash scripts/r10/base.sh old03 old05 m03 m18 m15 m19     -> /workspace/dub/r10/runs/<id>/, composites in artifacts/r4m/r9f_<id>{,stab}
set -u
cd /workspace/dub/dubai; . scripts/env.sh >/dev/null; export DUB_ROOT=/workspace/dub/dubai DUB_ALLOW_NC=1
B=/workspace/dub/artifacts/r4m; R=/workspace/dub/r10/runs; M=/workspace/dub/inputs/mamakris; T=/workspace/dub/inputs/test_videos_raw; mkdir -p $R
declare -A F=([m19]=$M/story-1774250832175-746471562.mp4 [m03]=$M/story-1774249919482-367803614.MOV [m15]=$M/story-1774250573471-666527943.MP4
  [m18]=$M/story-1774250817106-616905094.mp4 [old03]=$T/03.mp4 [old05]=$T/05.mp4)
FBA="$(cat reports/experiments/r8-face/fb7.args)"
for id in "$@"; do
  [ -f $R/$id/${id}_DabAI_r9_demucs.mp4 ] || { rm -rf $R/$id; E2E_FX=1 E2E_METRICS=off ID=$id bash scripts/r9/e2e.sh ${F[$id]} $R/$id > $R/$id.out 2>&1; }
  grep -E "R9_E2E_DONE" $R/$id.out | tail -1
  export DUB_INPUTS=$R/$id/work/in
  [ -f $B/r9f_${id}stab/${id}_r9f_${id}stab.mp4 ] || flock /workspace/dub/gpu.lock env METRICS=off SKIP_SCORE=1 TAG=r9f_${id}stab SRC_TAG=r9g_$id CFD=self CFD_LABEL=ia04 CFD_ARGS="--inner-alpha 0.4" \
    SR=srt1 SR_EXTRA="--batch 8 --mask down2" FB=fb7 FB_ARGS="$FBA" REP=1 STAB=1 REP_LABEL=stab bash scripts/r7/r7_m7_comp.sh $id > $R/$id.stab.log 2>&1
  echo "== $id base done $(date +%T) fx=$(ls $B/r9f_$id/${id}_r9f_$id.mp4 2>/dev/null) stab=$(ls $B/r9f_${id}stab/${id}_r9f_${id}stab.mp4 2>/dev/null)"
done
echo R10_BASE_DONE "$@"
