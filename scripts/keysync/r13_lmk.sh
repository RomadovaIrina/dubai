#!/bin/bash
# r13 speed lever: landmarks off the critical path (KS_LMK_CACHE: the first pass of each video fills the cache, the second reads it = the prod case
# where landmarks are computed during TTS / separation). f_s8 config otherwise. Eval on q18 + m03.
cd /workspace/dub/keysync; export KS_LMK_CACHE=/workspace/dub/keysync/fast/lmk_cache
IDS="q18 q18 m03 m03" KS_NOEVAL=1 KS_DC=2:0 bash ks_fast.sh f_s8lmk --cfg _s8 --vae-half --compile unet
grep -a -h KS_JOB fast/f_s8lmk/run.log | sed -E "s/\"peak_vram_gb.*//" | cut -c1-400
for i in q18 m03; do bash ks_eval.sh f_s8lmk $i | grep KS_EVAL; done; echo LMK_DONE
