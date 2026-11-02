#!/bin/bash
cd /workspace/dub/keysync; until grep -q ARM_STEPS_DONE arm_steps.log; do sleep 20; done
bash ks_mkgen.sh qo3 ksga_qo3 /workspace/dub/keysync/runs/qo3/out/qo3_qo3.mp4 --cm 12 && REP= bash ks_comp.sh qo3 ksfa_qo3 ksga_qo3 && bash ks_score1.sh qo3 /workspace/dub/inputs/test_videos_raw/03.mp4 ksfa_qo3; echo ARM_QO3A_DONE
