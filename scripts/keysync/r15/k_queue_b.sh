#!/bin/bash
# r15 K queue b: TRT engines -> the TRT arm -> TRT + batched VAE decode -> (evals || the mamakris dump) -> the LCM-LoRA training
K=/workspace/dub/keysync; cd $K
bash k_trt.sh > trt/k_trt.log 2>&1
if [ -s trt/eng/interp/full.plan ] && [ -s trt/eng/kf/full.plan ]; then
  KS_DC=2:0 bash k_arm2.sh trt --cfg _s8 --vae-half --compile unet --trt $K/trt/eng > trt/arm_trt.log 2>&1
  KS_DC=2:0 bash k_arm2.sh trt_vb --cfg _s8 --vae-half --compile unet --trt $K/trt/eng --vae-batch 14 > trt/arm_trt_vb.log 2>&1
fi
ls data/videos/tr_mk*.mp4 | while read f; do t=$(basename $f .mp4); [ -s dump/$t.pt ] || echo "$t $K/data/videos/$t.mp4 $K/data/audios/$t.wav"; done > dump/jobs_mk.txt
[ -s dump/jobs_mk.txt ] && bash k_dump.sh $K/dump/jobs_mk.txt > dump/k_dump_mk.out 2>&1
wait; while pgrep -f "k_eval.sh" > /dev/null; do sleep 10; done
bash k_train.sh r1 --val-tag tr_mk5 --minutes ${TRAIN_MIN:-95} --steps 6000 --rank 64 --lr 1e-4 > lcm_r1.out 2>&1
echo K_QUEUE_B_DONE
