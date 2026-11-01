#!/bin/bash
# r10: half-rate gate: face_only (E2E_FX look) with the e2e default face stack (_b) vs + LS_HALF_RATE=1 (_h), on the acct audio
cd /workspace/dub/dubai
export GEN_COMP=0 FAST_SRCPREP=1 FAST_FB_BG=1 FAST_ENC=ultrafast FAST_MPREUSE=1 FC_EXTRA=--pre-threads=12 LS_VAE_COMPILE=max-autotune-no-cudagraphs CFD_EXTRA=--threads=12
for id in m19 m03 03; do
  V=_b bash scripts/r9/face_only.sh $id 2>&1 | grep "==" 
  LS_HALF_RATE=1 V=_h bash scripts/r9/face_only.sh $id 2>&1 | grep "=="
done
echo HR_DONE
