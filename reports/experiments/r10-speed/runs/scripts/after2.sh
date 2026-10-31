#!/bin/bash
# r10: UNet leaf compile on m03 (cold = pays the compile, then warm + gate rows), then the m01 1080p60 E2E (default stack)
while pgrep -f "face_only[.]sh m03" >/dev/null; do sleep 15; done
cd /workspace/dub/dubai
export GEN_COMP=0 FAST_SRCPREP=1 FAST_FB_BG=1 FAST_ENC=ultrafast FAST_MPREUSE=1 FC_EXTRA=--pre-threads=12 LS_VAE_COMPILE=max-autotune-no-cudagraphs CFD_EXTRA=--threads=12
LS_UNET_COMPILE=default METRICS=off V=_uc0 bash scripts/r9/face_only.sh m03 2>&1 | grep "== timing"
LS_UNET_COMPILE=default V=_uc bash scripts/r9/face_only.sh m03 2>&1 | grep "== timing\|metrics done"
echo UC_DONE
unset GEN_COMP FAST_SRCPREP FAST_FB_BG FAST_ENC FAST_MPREUSE FC_EXTRA LS_VAE_COMPILE CFD_EXTRA
ID=m01 E2E_METRICS=off bash scripts/r9/e2e.sh /workspace/dub/inputs/mamakris/story-1773579769284-320619146.MP4 /workspace/dub/r10/acct/m01 > /workspace/dub/r10/acct/m01.out 2>&1
tail -n 3 /workspace/dub/r10/acct/m01.out
echo M01_DONE
