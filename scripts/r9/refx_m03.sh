#!/bin/bash
# m03 fx composites again with the lateral-only lip-colour detector, under new names (REP_LABEL=rep3, tags ...fx3): the first ones used the 2-D offset + spread rule
cd /workspace/dub/dubai; . scripts/env.sh >/dev/null; export DUB_ALLOW_NC=1 METRICS=off DUB_ROOT=/workspace/dub/dubai DUB_INPUTS=/workspace/dub/r9/runs/m03/work/in
FBA="$(cat reports/experiments/r8-face/fb7.args)"; B=/workspace/dub/artifacts/r4m; D=/workspace/dub/r9/diag; id=m03
for g in "" shift16L05 fixedL05; do
  W=$B/r9g_$id$g/work_${id}_r9g_$id$g; T=r9f_$id${g}fx3
  SKIP_SCORE=1 TAG=$T SRC_TAG=r9g_$id$g CFD=self CFD_LABEL=ia04 CFD_ARGS="--inner-alpha 0.4" SR=srt1 SR_EXTRA="--batch 8 --mask down2" FB=fb7 FB_ARGS="$FBA" REP=1 REP_LABEL=rep3 bash scripts/r7/r7_m7_comp.sh $id > $D/fx3_${id}$g.log 2>&1
  echo "== $T $(grep -h layer_repair $D/fx3_${id}$g.log | cut -c1-230) $(date +%T)"
  L3=$W/${id}_r9g_$id${g}_25fps_ls_cfd_ia04_srt1_rep3.mp4
  CUDA_VISIBLE_DEVICES= python $D/artefacts.py $id --tag-suffix ${g}fx3 --gen r9g_$id$g --l3 $L3 --out $D/art_$id${g}fx3.json 2>&1 | grep -v Warn | tail -1 > $D/art_$id${g}fx3.log
  python $D/teeth_hf.py $id ${g}fx3 r9g_$id$g $L3 > $D/teeth_$id${g}fx3.log 2>&1
done; echo REFX_DONE
