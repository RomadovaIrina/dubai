#!/bin/bash
until grep -q Q4UREF_DONE /workspace/dub/r10/after10.log; do sleep 30; done
cd /workspace/dub/dubai
export GEN_COMP=0 FAST_SRCPREP=1 FAST_FB_BG=1 FAST_ENC=ultrafast FAST_MPREUSE=1 FC_EXTRA=--pre-threads=12 LS_VAE_COMPILE=max-autotune-no-cudagraphs CFD_EXTRA=--threads=12
LS_UNET_COMPILE=default LS_UNET_COMPILE_DYNAMIC=1 V=_ucd bash scripts/r9/face_only.sh m03 2>&1 | grep "== timing\|metrics done"
sleep 900; python3 scripts/r9/gate_cmp.py --base /workspace/dub/r9/face/m03_b --cand /workspace/dub/r9/face/m03_ucd 2>&1 | tail -3
echo UCD_DONE
