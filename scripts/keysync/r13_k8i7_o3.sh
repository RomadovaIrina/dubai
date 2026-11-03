#!/bin/bash
# r13: k8i7 (keyframes 8, interpolation 7 steps) + landmark cache on old03 (the decisive flash / sync check; m18 / m03 were mixed)
cd /workspace/dub/keysync; export KS_LMK_CACHE=/workspace/dub/keysync/fast/lmk_cache KS_NOEVAL=1 KS_DC=2:0
IDS="qo3" bash ks_fast.sh f_k8i7_o3 --cfg _k8i7 --vae-half --compile unet
grep -a -h KS_JOB fast/f_k8i7_o3/run.log | sed -E "s/\"stages_s\": \{[^}]*\}, //; s/\"peak_vram_gb.*//" | cut -c1-200
bash ks_eval.sh f_k8i7_o3 qo3 | grep KS_EVAL; echo K8I7O3_DONE
