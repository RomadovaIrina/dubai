#!/bin/bash
until grep -q QBENCH_DONE /workspace/dub/r10/qbench.log; do sleep 30; done
cd /workspace/dub/dubai
export GEN_COMP=0 FAST_SRCPREP=1 FAST_FB_BG=1 FAST_ENC=ultrafast FAST_MPREUSE=1 FC_EXTRA=--pre-threads=12 LS_VAE_COMPILE=max-autotune-no-cudagraphs CFD_EXTRA=--threads=12
LS_PROFILE=/workspace/dub/r10/ls_profile_m03.txt METRICS=off V=_prof bash scripts/r9/face_only.sh m03 2>&1 | grep "== timing" | cut -c1-200
echo PROF_DONE
unset GEN_COMP FAST_SRCPREP FAST_FB_BG FAST_ENC FAST_MPREUSE FC_EXTRA LS_VAE_COMPILE CFD_EXTRA
ID=m01 E2E_METRICS=off bash scripts/r9/e2e.sh /workspace/dub/inputs/mamakris/story-1773579769284-320619146.MP4 /workspace/dub/r10/acct/m01 > /workspace/dub/r10/acct/m01.out 2>&1
tail -3 /workspace/dub/r10/acct/m01.out
echo M01_DONE
