#!/bin/bash
# r13 old03 quality items on top of cfhf (v1 base, no tsmooth, the same as the clips the viewer has): ff = + ks_fix --flash-fix 4 (single-frame mouth
# flash repair), ffl = ff + --lip-chroma 0.7 (the pale / vanishing upper lip). Gate numbers via r13_colorfix.sh meas.
cd /workspace/dub/keysync; B="--skin-cm 12 --cm-robust 1 --guard-fade 10"
export KS_HF=3 KS_SHARP=0.3
CF_FIX="$B --flash-fix 4 --flash-report /workspace/dub/keysync/final/qo3_ff/flash_fix.json" bash r13_colorfix.sh ff
CF_FIX="$B --flash-fix 4 --lip-chroma 0.7" bash r13_colorfix.sh ffl
echo FLASH_ALL_DONE
