#!/bin/bash
# r13 speed levers on top of f_s8 + KS_LMK_CACHE (landmarks off the critical path): SageAttention 1.0.6 (KS_SAGE=1), interpolation 6 / 7 steps
# with keyframes kept at 8 (k8i6 / k8i7). q18 twice (the first pays the compile), then m03; eval q18 + m03.
cd /workspace/dub/keysync; export KS_LMK_CACHE=/workspace/dub/keysync/fast/lmk_cache KS_NOEVAL=1 KS_DC=2:0
while ! grep -q LMK2_DONE r13_lmk2.log; do sleep 30; done
arm() {  # <arm> <cfg> [env...]
  A=$1; C=$2; shift 2
  env "$@" IDS="q18 q18 m03" bash ks_fast.sh $A --cfg $C --vae-half --compile unet
  grep -a -h KS_JOB fast/$A/run.log | sed -E "s/\"stages_s\": \{[^}]*\}, //; s/\"peak_vram_gb.*//" | sed "s/^/$A /" | cut -c1-200
  grep -a "Error" fast/$A/run.log | tail -2 | cut -c1-200
  for i in q18 m03; do bash ks_eval.sh $A $i | grep KS_EVAL; done
}
arm f_sage _s8 KS_SAGE=1
arm f_k8i7 _k8i7
arm f_k8i6 _k8i6
echo SPEED2_DONE
