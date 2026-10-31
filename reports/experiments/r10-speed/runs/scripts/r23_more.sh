#!/bin/bash
until grep -q OSGATE_DONE /workspace/dub/r10/acct/os_gate.log; do sleep 20; done
cd /workspace/dub/dubai
export GEN_COMP=0 FAST_SRCPREP=1 FAST_FB_BG=1 FAST_ENC=ultrafast FAST_MPREUSE=1 FC_EXTRA=--pre-threads=12 LS_VAE_COMPILE=max-autotune-no-cudagraphs CFD_EXTRA=--threads=12
for id in m19 03; do LS_RATE=2/3 V=_r23 bash scripts/r9/face_only.sh $id 2>&1 | grep "=="; done
echo R23_DONE
