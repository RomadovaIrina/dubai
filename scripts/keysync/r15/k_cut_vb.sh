#!/bin/bash
# stop the trt_vb arm after q18 (wu + q18): the batched VAE decode is an exact-math lever, measured on m18; the combined arm measures all 3
cd /workspace/dub/keysync
until [ -s fast/trt_vb/run.log ] && [ $(grep -a -c KS_JOB fast/trt_vb/run.log) -ge 2 ]; do sleep 5; done
pkill -f "ks_fast.py --jobs /workspace/dub/keysync/fast/trt_vb/jobs.txt" && echo "CUT trt_vb after q18 $(date)"
