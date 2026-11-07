#!/bin/bash
# r16 L on r16-k1: after the dump -> evals of the current student and jit10_s776 (m03a, m18a, o03a) -> spike + boundary run from s776.
R=/workspace/dub/r16l; L=$R/repo/scripts/keysync/r16/l
while ! grep -q DUMP_DONE $R/dump/dump.log; do sleep 20; done
bash $L/l_eval.sh /workspace/dub/r16in/lora/lcm_r1_lora_fp16.pt base $R/eval/jobs_a.txt > $R/eval/base.log 2>&1 &
sleep 120   # base render holds the lock first; its CPU metrics overlap the next render
bash $L/l_eval.sh $R/runs/jit10/lora_step_00776.pt jit10_s776 $R/eval/jobs_a.txt > $R/eval/jit10_s776.log 2>&1 &
sleep 120
mkdir -p $R/runs/spk1; cp $R/runs/jit10/lora_meta.json $R/runs/spk1/
bash $L/l_drive.sh spk1 6 2 --minutes 12 --lam 10 --spike-top 0.05 --pair 0.5 --swap 0.5 --lr 5e-5 --init $R/runs/jit10/lora_step_00776.pt > $R/runs/spk1.drive.log 2>&1
wait; echo QUEUE_DONE
