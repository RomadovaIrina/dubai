#!/bin/bash
# r16 K1 step 0: KeySync fast2c on the 40 ms delayed dub
cd /workspace/dub/keysync; until grep -qE "TRT_ALL_DONE|Traceback" trt_build.log; do sleep 10; done; grep -q TRT_ALL_DONE trt_build.log || { echo TRT_FAIL; exit 1; }
cd code; . ../venv/bin/activate
KS_DC=2:0 KS_LMK_CACHE=/workspace/dub/keysync/lmk flock /workspace/dub/gpu.lock python ../ks_fast.py --jobs ../r16/jobs_d40.txt --out ../r16/out_d40 \
  --cfg _s8 --vae-half --compile unet --trt ../trt/eng_s --interp-lora /workspace/dub/r16in/lora/lora_pick.pt --interp-steps 2 --vae-batch 8 --vae-compile
echo RUN_D40_DONE
