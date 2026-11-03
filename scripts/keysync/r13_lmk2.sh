#!/bin/bash
# r13 landmarks-off-path lever on old03 (qo3 twice: fill, then read the cache) + eval
cd /workspace/dub/keysync; export KS_LMK_CACHE=/workspace/dub/keysync/fast/lmk_cache
IDS="qo3 qo3" KS_NOEVAL=1 KS_DC=2:0 bash ks_fast.sh f_s8lmk_o3 --cfg _s8 --vae-half --compile unet
grep -a -h KS_JOB fast/f_s8lmk_o3/run.log | sed -E "s/\"peak_vram_gb.*//" | cut -c1-400
bash ks_eval.sh f_s8lmk_o3 qo3 | grep KS_EVAL; echo LMK2_DONE
