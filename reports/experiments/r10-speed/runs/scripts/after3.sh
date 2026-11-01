#!/bin/bash
# r10: UNet leaf compile re-test with raised dynamo limits (cold _uc1, warm _uc2 + gate), after m01
until grep -q M01_DONE /workspace/dub/r10/after2.log; do sleep 30; done
cd /workspace/dub/dubai
export GEN_COMP=0 FAST_SRCPREP=1 FAST_FB_BG=1 FAST_ENC=ultrafast FAST_MPREUSE=1 FC_EXTRA=--pre-threads=12 LS_VAE_COMPILE=max-autotune-no-cudagraphs CFD_EXTRA=--threads=12
LS_UNET_COMPILE=default METRICS=off V=_uc1 bash scripts/r9/face_only.sh m03 2>&1 | grep "== timing"
LS_UNET_COMPILE=default V=_uc2 bash scripts/r9/face_only.sh m03 2>&1 | grep "== timing\|metrics done"
echo UC2_DONE
