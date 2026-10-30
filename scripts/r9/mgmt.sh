#!/bin/bash
# r9 management set: <id>_{r9fix,fxlite,fxstab}_{demucs,spleeter}.mp4 in /workspace/dub/outbox/mgmt_r9/
#   m03 / m18: the existing composites; old03 / old05: full E2E (generate once, plain r9 composite) + 3 composites; then gate scoring of every variant.
set -u
cd /workspace/dub/dubai; . scripts/env.sh >/dev/null; export DUB_ROOT=/workspace/dub/dubai DUB_ALLOW_NC=1
A=/workspace/dub/artifacts/r4a; B=/workspace/dub/artifacts/r4m; D=/workspace/dub/r9/diag; O=/workspace/dub/outbox/mgmt_r9; mkdir -p $O; FBA="$(cat reports/experiments/r8-face/fb7.args)"
mux() {   # mux <id> <variant> <composite tag>
  local id=$1 v=$2 T=$3; for b in demucs:$A/r9_${id}_htd/$id/final_mix_aac.m4a spleeter:$A/r9_$id/$id/final_mix_aac.m4a; do
    ffmpeg -loglevel error -y -i $B/$T/${id}_$T.mp4 -i ${b#*:} -map 0:v:0 -map 1:a:0 -c copy -movflags +faststart $O/${id}_${v}_${b%%:*}.mp4; done; echo "== muxed $id $v ($T)"; }
comp() {  # comp <id> <tag> <extra env...>
  local id=$1 T=$2; shift 2; export DUB_INPUTS=/workspace/dub/r9/runs/$id/work/in
  [ -f $B/$T/${id}_$T.mp4 ] || flock /workspace/dub/gpu.lock env METRICS=off SKIP_SCORE=1 TAG=$T SRC_TAG=r9g_$id CFD=self CFD_LABEL=ia04 CFD_ARGS="--inner-alpha 0.4" SR=srt1 SR_EXTRA="--batch 8 --mask down2" FB=fb7 FB_ARGS="$FBA" REP=1 "$@" bash scripts/r7/r7_m7_comp.sh $id > $D/mgmt_$T.log 2>&1
  echo "== comp $T $(grep -h layer_repair $D/mgmt_$T.log 2>/dev/null | cut -c1-140)"; }
# 1. m03 / m18: existing composites
mux m03 r9fix r9f_m03fx3; mux m03 fxlite r9f_m03fxlite; mux m03 fxstab r9f_m03fxstab
mux m18 r9fix r9f_m18fx;  mux m18 fxlite r9f_m18fxlite; mux m18 fxstab r9f_m18fxstab
# 2. the old test videos: full E2E (production only), then 3 composites each
for n in 03 05; do id=old$n
  [ -f /workspace/dub/r9/runs/$id/${id}_DabAI_r9_demucs.mp4 ] || E2E_METRICS=off ID=$id bash scripts/r9/e2e.sh /workspace/dub/inputs/test_videos_raw/$n.mp4 /workspace/dub/r9/runs/$id > /workspace/dub/r9/runs/$id.out 2>&1
  grep -E "SUCCESS|PARTIAL|ERROR" /workspace/dub/r9/runs/$id.out | tail -1
  comp $id r9f_${id}fx; comp $id r9f_${id}fxlite REP_LABEL=lite REP_ARGS="--spikes off"; comp $id r9f_${id}fxstab STAB=1 REP_LABEL=stab
  mux $id r9fix r9f_${id}fx; mux $id fxlite r9f_${id}fxlite; mux $id fxstab r9f_${id}fxstab
done
echo MGMT_FILES_DONE
# 3. gate rows of every variant (skip what is scored already)
mkdir -p /workspace/dub/artifacts/r6f/lm
for id in m03 m18 old03 old05; do export DUB_INPUTS=/workspace/dub/r9/runs/$id/work/in; src=$(ls $DUB_INPUTS/$id.* | head -1)
  [ -f /workspace/dub/artifacts/r6f/lm/${id}_src_lm.npz ] || python scripts/r5/r5_srclm.py --src $src --out /workspace/dub/artifacts/r6f/lm/${id}_src_lm.npz > /dev/null 2>&1
  for T in $( [ $id = m03 ] && echo r9f_m03fx3 || echo r9f_${id}fx ) r9f_${id}fxlite r9f_${id}fxstab; do
    [ -f $B/$T/${id}_$T.score.json ] || flock /workspace/dub/gpu.lock python scripts/r4/r4_gate.py score --label $T --dir $B/$T --ids $id > $D/score_$T.log 2>&1
    [ -f $D/flash_$T.json ] || python scripts/quality/r3_flash.py --video $B/$T/${id}_$T.mp4 --src $src --lm $B/r9g_$id/timing/${id}_source_lm_F.npz --out $D/flash_$T.json > /dev/null 2>&1
    [ -f /workspace/dub/artifacts/r6f/twitch/$T/$id.twitch.json ] || { mkdir -p /workspace/dub/artifacts/r6f/twitch/$T; python scripts/r5/r5_twitch.py score --video $B/$T/${id}_$T.mp4 --src $src --lm /workspace/dub/artifacts/r6f/lm/${id}_src_lm.npz --manifest $B/$T/${id}_$T.manifest.json --out /workspace/dub/artifacts/r6f/twitch/$T/$id.twitch.json > /dev/null 2>&1; }
    echo "SCORED $T $(date +%T)"
  done
done
echo MGMT_DONE
