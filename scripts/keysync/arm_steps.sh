#!/bin/bash
# KeySync m18 step arms (6 / 8 steps) through the composite, after the old03 arm
cd /workspace/dub/keysync; S=/workspace/dub/inputs/mamakris/story-1774250817106-616905094.mp4
until grep -q KS_SCORE1_DONE arm_ksfn_qo3.log; do sleep 20; done
for s in s6 s8; do bash ks_mkgen.sh q18 ksg${s}_q18 /workspace/dub/keysync/runs/q18_$s/out/q18_q18.mp4 --cm 12 && REP= bash ks_comp.sh q18 ksfn${s}_q18 ksg${s}_q18 && bash ks_score1.sh q18 $S ksfn${s}_q18; done; echo ARM_STEPS_DONE
