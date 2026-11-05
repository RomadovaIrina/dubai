#!/bin/bash
# r15 K round 2: the "KeySync fast" candidates = the 4- / 2-step LCM student (TRT, LoRA merged) + TRT keyframe UNet + the compiled VAE decoder (8 frames / call)
K=/workspace/dub/keysync; L=$K/lcm/r1/lora_pick.pt; cd $K
A="--cfg _s8 --vae-half --compile unet --trt $K/trt/eng_s --interp-lora $L --vae-batch 8 --vae-compile"
KS_DC=2:0 bash k_arm2.sh fast4c $A --interp-steps 4 > lcm/arm_fast4c.log 2>&1
KS_DC=2:0 bash k_arm2.sh fast2c $A --interp-steps 2 > lcm/arm_fast2c.log 2>&1
while pgrep -f "[k]_eval.sh" > /dev/null; do sleep 10; done
echo K_FINAL_ARMS_DONE
