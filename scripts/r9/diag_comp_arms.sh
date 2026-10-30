#!/bin/bash
# composite-only arms on m03 (the shipped generator layers r9g_m03): REP + inner-mouth detail weight, then + no extra lip sharpen
cd /workspace/dub/dubai; . scripts/env.sh >/dev/null; export DUB_ALLOW_NC=1 METRICS=off DUB_INPUTS=/workspace/dub/r9/runs/m03/work/in DUB_ROOT=/workspace/dub/dubai
FBA="$(cat reports/experiments/r8-face/fb7.args)"; W=/workspace/dub/artifacts/r4m/r9g_m03/work_m03_r9g_m03; D=/workspace/dub/r9/diag
SKIP_SCORE=1 TAG=r9f_m03repia SRC_TAG=r9g_m03 CFD=self CFD_LABEL=ia04 CFD_ARGS="--inner-alpha 0.4" SR=srt1 SR_EXTRA="--batch 8 --mask down2" FB=fb7 FB_ARGS="$FBA" REP=1 bash scripts/r7/r7_m7_comp.sh m03 > $D/comp_repia.log 2>&1
SKIP_SCORE=1 TAG=r9f_m03repians SRC_TAG=r9g_m03 CFD=self CFD_LABEL=ia04 CFD_ARGS="--inner-alpha 0.4" SR=srt1 SR_EXTRA="--batch 8 --mask down2" FB=fb7 FB_ARGS="$FBA" REP=1 EXTRA="--tex-tau 1000 --tex-sigma 2.5 --sharpen-lips-max 0" bash scripts/r7/r7_m7_comp.sh m03 > $D/comp_repians.log 2>&1
for t in repia repians; do CUDA_VISIBLE_DEVICES= python $D/artefacts.py m03 --tag-suffix $t --gen r9g_m03 --l3 $W/m03_r9g_m03_25fps_ls_cfd_ia04_srt1_rep.mp4 --out $D/art_m03$t.json 2>&1 | grep -v Warn | tail -1
  python $D/teeth_hf.py m03 $t r9g_m03 $W/m03_r9g_m03_25fps_ls_cfd_ia04_srt1_rep.mp4 2>&1 | tail -1; done
echo COMP_ARMS_DONE
