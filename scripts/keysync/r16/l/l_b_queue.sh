#!/bin/bash
# r16 L day 2: B (per-person keyframe LoRA) prepared on m18 / old03 + the colour-neutral variant (also on m03, vs adapt_m03s = B2).
# Full-length raw-crop renders (fast2 + the current student lcm_r1); low GPU priority: every GPU step under flock -o gpu.lock.
#   l_b_queue.sh   -> r16l/adapt/<v>_<arm>/kf_lora.pt, r16l/eval/B_<v>_<arm>/ (+ base), colour json; summary in r16l/eval/B_summary.txt
R=/workspace/dub/r16l; K=/workspace/dub/keysync; L=$R/repo/scripts/keysync/r16/l; STU=${STU:-/workspace/dub/r16in/lora/lcm_r1_lora_fp16.pt}
KF=$(find $K/trt -path "*eng/kf/full.plan" 2>/dev/null | head -1); mkdir -p $R/trt_kf $R/trt_none; [ -n "$KF" ] && ln -sfn $(dirname $KF) $R/trt_kf/kf
echo "kf engine: ${KF:-NONE (torch keyframes)}"
adapt() {   # adapt <video> <arm> <extra args>
  local v=$1 a=$2; shift 2; local O=$R/adapt/${v}_$a; mkdir -p $O; [ -s $O/kf_lora.pt ] && return 0
  (cd $K/code && . ../venv/bin/activate && flock -o /workspace/dub/gpu.lock nice python $L/ks_adapt.py --video $R/data/videos/${v}_src.mp4 --audio $R/data/audios/${v}_src.wav \
     --out $O --seconds 240 --lr 1e-3 "$@" > $O/adapt.log 2>&1); grep ADAPT_TIME $O/adapt.log | sed "s/^/$v $a /"; }
evalv() {   # evalv <video> <arm> (arm=base: no keyframe LoRA, kf TRT)
  local v=$1 a=$2; local N=B_${v}_$a; grep "^$v " $R/eval/jobs_full.txt > $R/eval/jobs_$v.txt
  if [ $a = base ]; then FLOW_T1=99 bash $L/l_eval.sh $STU $N $R/eval/jobs_$v.txt > $R/eval/$N.log 2>&1
  else KS_KF_LORA=$R/adapt/${v}_$a/kf_lora.pt TRTDIR=$R/trt_none FLOW_T1=99 bash $L/l_eval.sh $STU $N $R/eval/jobs_$v.txt > $R/eval/$N.log 2>&1; fi
  /venv/dabai/bin/python $L/l_colour.py --gen $R/eval/$N/$v/${v}_$v.mp4 --src $R/data/videos/${v}_src.mp4 --src-lm $R/data/videos/$v.mp.npz --out $R/eval/$N/$v.colour.json > /dev/null
  echo "$N $(python3 $L/l_agg.py $R/eval/$N | sed -E 's/flow \{[^}]*\} //; s/.top_t.: \[[^]]*\]//; s/bnd13.*black/black/') colour $(cat $R/eval/$N/$v.colour.json)" | tee -a $R/eval/B_summary.txt; }
# m03: the colour-neutral variant vs B2 (adapt_m03s = prior 0, from the laptop)
mkdir -p $R/adapt/m03_p0; [ -s $R/adapt/m03_p0/kf_lora.pt ] || cp /workspace/dub/r16in/ckpt/adapt_m03s/kf_lora.pt /workspace/dub/r16in/ckpt/adapt_m03s/lora_meta.json $R/adapt/m03_p0/
adapt m03 p1 --prior 1.0
for v in m18 old03; do adapt $v p0; adapt $v p1 --prior 1.0; done
for v in m03 m18 old03; do for a in base p0 p1; do evalv $v $a; done; done
echo B_QUEUE_DONE
