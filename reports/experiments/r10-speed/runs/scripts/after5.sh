#!/bin/bash
# r10: m01 (1080p60) composite OOM (fps_composite rc 137 at the 63.8 GB cgroup limit): re-run ONLY the composite (layer caches reused), peak RSS, without / with --pre-threads
until grep -q UC2_DONE /workspace/dub/r10/after4.log; do sleep 30; done
cd /workspace/dub/dubai; . scripts/env.sh >/dev/null; export DUB_ROOT=/workspace/dub/dubai DUB_ALLOW_NC=1 METRICS=off
export DUB_INPUTS=/workspace/dub/r10/acct/m01/work/in FAST_ENC=ultrafast FAST_FB_BG=1 REP=1 CFD_LABEL=ia04 CFD_ARGS="--inner-alpha 0.4 --threads=12"
for fx in nopre:"" pre:"--pre-threads=12"; do lab=${fx%%:*}; FC_EXTRA="${fx#*:}"; export FC_EXTRA
  python3 /workspace/dub/r10/maxrss.py env SKIP_SCORE=1 TAG=r9f_m01$lab SRC_TAG=r9g_m01 CFD=self SR=srt1 SR_EXTRA="--batch 8 --mask down2" FB=fb7 FB_ARGS="$(cat reports/experiments/r8-face/fb7.args)" \
    bash scripts/r7/r7_m7_comp.sh m01 > /workspace/dub/r10/acct/m01_comp_$lab.log 2>&1
  echo "M01COMP $lab $(grep -c FAIL /workspace/dub/r10/acct/m01_comp_$lab.log) fails; $(grep 'Maximum resident\|Elapsed' /workspace/dub/r10/acct/m01_comp_$lab.log | tr '\n' ' ')"; ls /workspace/dub/artifacts/r4m/r9f_m01$lab/*.mp4 2>/dev/null
done
echo M01COMP_DONE
