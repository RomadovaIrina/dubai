#!/bin/bash
# r16 L (A): train in short locked chunks (other agents interleave), eval every E chunks; ckpt + metrics json -> r16l/ckpt/.
#   l_drive.sh <run> <chunks> <eval every E chunks> [ks_lcm_jit args]
R=/workspace/dub/r16l; K=/workspace/dub/keysync; N=$1; C=$2; E=$3; shift 3; D=$R/runs/$N; mkdir -p $D
for c in $(seq 1 $C); do
  [ -f $D/STOP ] && { echo STOP; break; }
  (cd $K/code && . $K/venv/bin/activate && flock /workspace/dub/gpu.lock python $R/repo/scripts/keysync/r16/l/ks_lcm_jit.py --dump $R/dump --out $D --resume "$@" >> $D/train.log 2>&1)
  grep -E "STEP|CHUNK_DONE|Error" $D/train.log | tail -2
  if [ $c -eq 1 ] || [ $((c % E)) -eq 0 ]; then
    s=$(ls $D/lora_step_*.pt | tail -1); st=$(basename $s .pt | sed 's/lora_step_0*//')
    bash $R/repo/scripts/keysync/r16/l/l_eval.sh $s ${N}_s$st $R/eval/jobs_a.txt > $R/eval/${N}_s$st.log 2>&1
    mkdir -p $R/ckpt/${N}_s$st; cp $s $D/lora_meta.json $R/ckpt/${N}_s$st/; cp $R/eval/${N}_s$st/metrics.json $R/ckpt/${N}_s$st/
    echo "EVAL ${N}_s$st"; tail -3 $R/eval/${N}_s$st.log
  fi
done
echo DRIVE_DONE
