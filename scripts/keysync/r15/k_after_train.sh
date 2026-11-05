#!/bin/bash
# r15 K: after the LCM training -> pick the checkpoint with the best 4-step latent PSNR vs the 8-step CFG teacher (val clip tr_mk5) -> k_queue2.sh
K=/workspace/dub/keysync; R=$K/lcm/${1:-r1}; cd $K
until grep -a -q -E "TRAIN_DONE|Traceback" $R/train.log 2>/dev/null; do sleep 30; done
sleep 60   # the final VAL line
L=$(python3 - $R <<'PY'
import json, sys, os
R = sys.argv[1]; best = None
for l in open(f"{R}/train.log", errors="ignore"):
    if l.startswith("VAL "):
        v = json.loads(l[4:]); s = v["step"]; f = f"{R}/lora_step_{s:05d}.pt"
        if s > 0 and os.path.exists(f) and (best is None or v["psnr_vs_teacher_4"] >= best[0]): best = (v["psnr_vs_teacher_4"], f)
print(best[1] if best else f"{R}/lora_last.pt")
PY
)
echo "PICKED $L"; cp $L $R/lora_pick.pt
bash k_queue2.sh $R/lora_pick.pt 4
[ -s trt/eng_s/interp/full.plan ] && KS_DC=2:0 bash k_arm2.sh fast2 --cfg _s8 --vae-half --compile unet --trt $K/trt/eng_s --interp-lora $R/lora_pick.pt --interp-steps 2 --vae-batch 4 > lcm/arm_fast2.log 2>&1
wait; while pgrep -f "[k]_eval.sh" > /dev/null; do sleep 10; done
echo K_AFTER_TRAIN_DONE
# appended 17:10: the TensorRT VAE decoder (interpolation model, per-frame math, 4 frames / call) + the combined 4-step arm with it
( cd code && . $K/venv/bin/activate && flock /workspace/dub/gpu.lock python $K/ks_trt_vae.py --video $K/data/videos/q18.mp4 --out $K/trt/vae --n 4 > $K/trt/make_vae.log 2>&1 )
grep -a -E "TRT_VAE_VERIFY|Traceback" trt/make_vae.log | cut -c1-600
[ -s trt/vae/dec.plan ] && [ -s trt/eng_s/interp/full.plan ] && KS_DC=2:0 bash k_arm2.sh fast4v --cfg _s8 --vae-half --compile unet --trt $K/trt/eng_s --trt-vae $K/trt/vae --interp-lora $R/lora_pick.pt --interp-steps 4 --vae-batch 4 > lcm/arm_fast4v.log 2>&1
wait; while pgrep -f "[k]_eval.sh" > /dev/null; do sleep 10; done
echo K_AFTER_TRAIN_DONE2
