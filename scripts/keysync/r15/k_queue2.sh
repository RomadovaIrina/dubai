#!/bin/bash
# r15 K queue 2 (after the training): the LCM student alone (4 / 2 steps), then the student's TensorRT engine (merged LoRA, no DeepCache)
# + the keyframe TRT engine + the batched VAE decode = the combined fast arm.   usage: bash k_queue2.sh <lora ckpt> [steps for the combined arm]
L=$1; S=${2:-4}; K=/workspace/dub/keysync; cd $K
KS_DC=2:0 bash k_arm2.sh lcm4 --cfg _s8 --vae-half --compile unet --interp-lora $L --interp-steps 4 > lcm/arm_lcm4.log 2>&1
KS_DC=2:0 bash k_arm2.sh lcm2 --cfg _s8 --vae-half --compile unet --interp-lora $L --interp-steps 2 > lcm/arm_lcm2.log 2>&1
mkdir -p trt/eng_s; ln -sfn $K/trt/eng/kf trt/eng_s/kf
( cd code && . $K/venv/bin/activate && flock /workspace/dub/gpu.lock python $K/ks_trt.py --cap $K/trt/cap --which interp --out $K/trt/eng_s --lora $L --no-dc > $K/trt/make_student.log 2>&1 )
grep -a -E "TRT_VERIFY|Traceback" trt/make_student.log | cut -c1-600
[ -s trt/eng_s/interp/full.plan ] && KS_DC=2:0 bash k_arm2.sh fast$S --cfg _s8 --vae-half --compile unet --trt $K/trt/eng_s --interp-lora $L --interp-steps $S --vae-batch 4 > lcm/arm_fast$S.log 2>&1
wait; while pgrep -f "[k]_eval.sh" > /dev/null; do sleep 10; done
echo K_QUEUE2_DONE
