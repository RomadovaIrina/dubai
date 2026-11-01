#!/bin/bash
# r10: generator keep-pattern variants on m03 (after hr_gate.sh), gated vs m03_b
until grep -q HR_DONE /workspace/dub/r10/acct/hr_gate.log; do sleep 20; done
cd /workspace/dub/dubai
export GEN_COMP=0 FAST_SRCPREP=1 FAST_FB_BG=1 FAST_ENC=ultrafast FAST_MPREUSE=1 FC_EXTRA=--pre-threads=12 LS_VAE_COMPILE=max-autotune-no-cudagraphs CFD_EXTRA=--threads=12
for v in 23:2/3 12:1/2; do LS_RATE=${v#*:} V=_r${v%%:*} bash scripts/r9/face_only.sh m03 2>&1 | grep "=="; done
echo RATE_DONE
