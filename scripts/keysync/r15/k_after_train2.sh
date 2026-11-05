#!/bin/bash
# r15 K: after the LCM training (replaces k_after_train.sh: the 6-segment val PSNR swings +-0.7 dB, so the FINAL checkpoint is used).
# Order = the key numbers first: the student's TRT engine -> fast4 (student 4 steps + TRT both UNets + batched decode) -> the TRT VAE decoder
# -> fast4v / fast2v (+ the TRT decode) -> lcm4 (the student alone under torch.compile = lever 2 measured separately).
K=/workspace/dub/keysync; R=$K/lcm/${1:-r1}; cd $K; . $K/venv/bin/activate
until grep -a -q -E "TRAIN_DONE|Traceback" $R/train.log 2>/dev/null; do sleep 30; done
L=$R/lora_last.pt; cp $L $R/lora_pick.pt; echo "PICKED $L"; L=$R/lora_pick.pt
mkdir -p trt/eng_s; ln -sfn $K/trt/eng/kf trt/eng_s/kf
( cd code && flock /workspace/dub/gpu.lock python $K/ks_trt.py --cap $K/trt/cap --which interp --out $K/trt/eng_s --lora $L --no-dc > $K/trt/make_student.log 2>&1 )
grep -a -E "TRT_VERIFY|Traceback" trt/make_student.log | cut -c1-600
A="--cfg _s8 --vae-half --compile unet --interp-lora $L"
[ -s trt/eng_s/interp/full.plan ] && KS_DC=2:0 bash k_arm2.sh fast4 $A --trt $K/trt/eng_s --interp-steps 4 --vae-batch 4 > lcm/arm_fast4.log 2>&1
( cd code && flock /workspace/dub/gpu.lock python $K/ks_trt_vae.py --video $K/data/videos/q18.mp4 --out $K/trt/vae --n 4 > $K/trt/make_vae.log 2>&1 )
grep -a -E "TRT_VAE_VERIFY|Traceback" trt/make_vae.log | cut -c1-600
V=""; [ -s trt/vae/dec.plan ] && V="--trt-vae $K/trt/vae"
[ -s trt/eng_s/interp/full.plan ] && KS_DC=2:0 bash k_arm2.sh fast4v $A --trt $K/trt/eng_s $V --interp-steps 4 --vae-batch 4 > lcm/arm_fast4v.log 2>&1
[ -s trt/eng_s/interp/full.plan ] && KS_DC=2:0 bash k_arm2.sh fast2v $A --trt $K/trt/eng_s $V --interp-steps 2 --vae-batch 4 > lcm/arm_fast2v.log 2>&1
KS_DC=2:0 bash k_arm2.sh lcm4 $A --interp-steps 4 > lcm/arm_lcm4.log 2>&1
while pgrep -f "[k]_eval.sh" > /dev/null; do sleep 10; done
echo K_AFTER_TRAIN2_DONE
