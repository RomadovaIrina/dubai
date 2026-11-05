#!/bin/bash
# r15 Z diagnostics on m18 (what drives the per-frame lip flicker?): CFG truncation on the teacher, the student; each + colour lock + F lip_flicker
cd /workspace/dub/dubai; . scripts/env.sh >/dev/null; CK=/workspace/dub/r15w/ckpt/r2_s4_step_01500_ema.pt
until grep -q RF_DONE /workspace/dub/r15w/rf_m18.log 2>/dev/null; do sleep 20; done
LS_CFG_TRUNC=3 V=q2ct3 bash scripts/r15/arm.sh m18 2>&1 | grep "^=="
R10_GEN_PY=scripts/r13/distill/student_gen.py R13_STUDENT_LORA=$CK R13_STUDENT_STEPS=4 LS_STRIDE=8 LS_OVERLAP=continue V=s4c8 bash scripts/r15/arm.sh m18 2>&1 | grep "^=="
for a in q2ct3 s4c8; do [ -s /workspace/dub/r15w/eval/${a}_m18/${a}_m18.mp4 ] || continue
  SRC=$a V=${a}lk LT_ARGS="--mode color" LT_LABEL=c bash scripts/r15/comp_variant.sh m18 2>&1 | grep "^==" | tail -1
  bash scripts/r15/zlf.sh m18 $a $a; bash scripts/r15/zlf.sh m18 ${a}lk $a; done
echo DIAG_DONE
