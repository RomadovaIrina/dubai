#!/bin/bash
cd /workspace/dub/dubai; . scripts/env.sh >/dev/null
export SPOTS="3.2:5.5 11.3:14.3 19.5:21.0 26.7:28.7"
bash scripts/r15/panel3.sh m03 q2d q2ts25 q2ts25lk "q2 prod (face units)" "CFG tsmooth 0.25" "tsmooth 0.25 + colour lock" 2>&1 | tail -1
bash scripts/r15/panel3.sh m03 q2d q2g15 q2g15lk "q2 prod (face units)" "guidance 1.5" "guidance 1.5 + colour lock" 2>&1 | tail -1
bash scripts/r15/panel3.sh m03 q2d q2g125 q2g125lk "q2 prod (face units)" "guidance 1.25" "guidance 1.25 + colour lock" 2>&1 | tail -1
echo PANELS_M03_DONE
