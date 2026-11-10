#!/bin/bash
# abl.sh ARM "ENV..." [videos]  : Q ablation, CPU re-paste of T's finished r16d jobs on a private COPY of the job dirs (T's dirs untouched),
# via T's own r16d_repost.sh (same code path as today's finals), then Q's detectors. KS_PLAIN_EXTRA (has spaces) comes from the env var ABL_EXTRA.
ARM=$1; ENVS=$2; shift 2; VIDS=${@:-m03 m18 old03 q01}
A=/workspace/dub/q_abl; W=$A/work_$ARM; O=$A/out/$ARM; mkdir -p $W/jobs $O $A/log
[ -d $W/json ] || cp -a /workspace/dub/r16d/json $W/json
for v in $VIDS; do
  ( [ -d $W/jobs/r16d_$v ] || cp -a /workspace/dub/r16d/jobs/r16d_$v $W/jobs/r16d_$v
    [ -n "${ABL_EXTRA:-}" ] && export KS_PLAIN_EXTRA="$ABL_EXTRA"
    R16D_WORK=$W R16D_OUT=$O R16D_ENV="$ENVS KS_PASTE_JOBS=${ABL_JOBS:-10}" bash /workspace/dub/t_tail/scripts/keysync/r16/r16d_repost.sh $v > $A/log/${ARM}_$v.log 2>&1
    M=/workspace/dub/r16d/jobs/r16d_$v/e2e/work/ks/master_25fps.mp4
    [ -s $O/$v.mp4 ] && (cd /workspace/dub/q && ./arm.sh abl_$ARM $v $M $O/$v.mp4 >> $A/SUMMARY.txt 2>&1)
    echo "$ARM $v done $(date +%T)" >> $A/done.txt ) &
done; wait
