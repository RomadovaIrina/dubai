#!/bin/bash
# fx = REP v2 (layer repair + lip-colour slide detector) + --inner-alpha 0.4, composite-only on each generator: shipped r9 / shift16L1 / fixedL1
cd /workspace/dub/dubai; . scripts/env.sh >/dev/null; export DUB_ALLOW_NC=1 METRICS=off DUB_ROOT=/workspace/dub/dubai
FBA="$(cat reports/experiments/r8-face/fb7.args)"; B=/workspace/dub/artifacts/r4m; A=/workspace/dub/artifacts/r4a; D=/workspace/dub/r9/diag
for id in m03 m19 m15; do export DUB_INPUTS=/workspace/dub/r9/runs/$id/work/in
  for g in "" shift16L05 fixedL05; do
    while [ ! -f $B/r9f_$id$g/${id}_r9f_$id$g.mp4 ]; do sleep 30; done   # the A/B queue is done with this work dir
    T=r9f_$id${g}fx; SKIP_SCORE=1 TAG=$T SRC_TAG=r9g_$id$g CFD=self CFD_LABEL=ia04 CFD_ARGS="--inner-alpha 0.4" SR=srt1 SR_EXTRA="--batch 8 --mask down2" FB=fb7 FB_ARGS="$FBA" REP=1 \
      bash scripts/r7/r7_m7_comp.sh $id > $D/fx_${id}$g.log 2>&1
    echo "== $T $(grep -h "layer_repair" $D/fx_${id}$g.log | cut -c1-220) $(date +%T)"
    ( W=$B/r9g_$id$g/work_${id}_r9g_$id$g; L3=$W/${id}_r9g_$id${g}_25fps_ls_cfd_ia04_srt1_rep2.mp4
      CUDA_VISIBLE_DEVICES= python $D/artefacts.py $id --tag-suffix ${g}fx --gen r9g_$id$g --l3 $L3 --out $D/art_$id${g}fx.json 2>&1 | grep -v Warn | tail -1 > $D/art_$id${g}fx.log
      python $D/teeth_hf.py $id ${g}fx r9g_$id$g $L3 > $D/teeth_$id${g}fx.log 2>&1 ) < /dev/null &
  done
done
wait; echo FX_DONE
