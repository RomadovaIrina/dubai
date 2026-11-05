#!/bin/bash
# r15 K queue: baseline evals || TRT engines -> the TRT arm (ks_fast) -> its evals || the mamakris dump -> the LCM-LoRA training
K=/workspace/dub/keysync; cd $K
until grep -q KSF_DONE k_base.log; do sleep 20; done
(for i in q18 m03 qo3; do bash k_eval.sh prod $i; done > fast/prod/evals.log 2>&1) &
bash k_trt.sh > trt/k_trt.log 2>&1
if [ -s trt/eng/interp/full.plan ] && [ -s trt/eng/kf/full.plan ]; then
  IDS="wu q18 m03 qo3" KS_NOEVAL=1 KS_DC=2:0 KS_LMK_CACHE=$K/lmk bash ks_fast.sh trt --cfg _s8 --vae-half --compile unet --trt $K/trt/eng > trt/arm_trt.log 2>&1
  (for i in q18 m03 qo3; do bash k_eval.sh trt $i; done > fast/trt/evals.log 2>&1) &
fi
wait
bash k_train.sh r1 --val-tag tr_mk5 --minutes ${TRAIN_MIN:-105} --steps 6000 --rank 64 --lr 1e-4 > lcm_r1.out 2>&1
echo K_QUEUE1_DONE
