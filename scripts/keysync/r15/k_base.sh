#!/bin/bash
# r15 K baseline = the production KeySync face stage (r13 default): f_s8 (8 steps both models, full 3-way CFG) + KS_DC=2:0 + fp16 VAE + compiled UNets
# + KS_LMK_CACHE; q18 first (compile warm-up; also captures the UNet inputs for TensorRT), then m03, qo3, q18 again (warm).
K=/workspace/dub/keysync; cd $K
IDS="q18 m03 qo3 q18" KS_NOEVAL=1 KS_DC=2:0 KS_LMK_CACHE=$K/lmk KS_CAPTURE=$K/trt/cap bash ks_fast.sh prod --cfg _s8 --vae-half --compile unet
