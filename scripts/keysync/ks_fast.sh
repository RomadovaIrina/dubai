#!/bin/bash
# ks_fast.sh <arm> [ks_fast.py args]: the resident KeySync worker on IDS (default "q18 qo3") under gpu.lock, NVML-sampled -> /workspace/dub/keysync/fast/<arm>/
# env passes through (KS_DC=N:B ...); then ks_eval.sh per video
A=$1; shift; K=/workspace/dub/keysync; O=$K/fast/$A; mkdir -p $O
: > $O/jobs.txt; for i in ${IDS:-q18 qo3}; do echo "$i $K/data/videos/$i.mp4 $K/data/audios/$i.wav" >> $O/jobs.txt; done
cd $K/code; . $K/venv/bin/activate
flock /workspace/dub/gpu.lock bash -c "/venv/main/bin/python /workspace/dub/dubai/scripts/r10/gpu_sampler.py --out $O/nvml.jsonl 2>/dev/null & SP=\$!; python $K/ks_fast.py --jobs $O/jobs.txt --out $O/out $* > $O/run.log 2>&1; kill \$SP"
grep -E "^\{|KS_JOB" $O/run.log
[ -n "$KS_NOEVAL" ] || for i in ${IDS:-q18 qo3}; do bash $K/ks_eval.sh $A $i | grep KS_EVAL; done; echo KSF_DONE $A
