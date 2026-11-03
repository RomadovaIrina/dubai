#!/bin/bash
# r13 KS speed arms on the full-CFG stack f_s8 (r12 HANDOFF NEXT 1): reproduce f_s8, then decode fix, FP8 PerTensor, 6 steps; then the 10-step f_base
# (the gate reference, in case the regenerated dub voice differs from r12's). m18 / m03 / old03. The GPU part runs back to back; each arm's eval
# (ks_eval.sh: AV shift, alpha 0.8 smoothing, SyncNet, flashes, jerk) runs in the background meanwhile (its SyncNet waits for gpu.lock).
cd /workspace/dub/keysync
until grep -q KS_SETUP_DONE setup_ks.log && grep -q R13_PREP_DONE r13_prep.log && grep -q R13_E2E_DONE r13_e2e.log; do sleep 30; done
S="--cfg _s8 --vae-half --compile unet"; export IDS="q18 q18 m03 qo3" KS_NOEVAL=1
ev() { ( for i in q18 m03 qo3; do bash ks_eval.sh $1 $i | grep KS_EVAL; done ) & }
KS_DC=2:0 bash ks_fast.sh f_s8 $S; ev f_s8
KS_DC=2:0 bash ks_fast.sh f_s8dec $S --vae-cl --cudnn-bench --decoding-t 4; ev f_s8dec
KS_DC=2:0 bash ks_fast.sh f_s8fp8 $S --fp8; ev f_s8fp8
KS_DC=2:0 bash ks_fast.sh f_s6 --cfg _s6 --vae-half --compile unet; ev f_s6
IDS="q18 m03 qo3" bash ks_fast.sh f_base; ev f_base
wait; echo R13_ARMS_DONE
