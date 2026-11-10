#!/bin/bash
# the ladder: P0 oval (08.11-style) -> +tight -> +parse -> (+cm-shadow side arm) -> +occl -> +cap -> +pause ; +tail = today's final (already rendered)
cd /workspace/dub/q_abl
OFF="KS_OCCL=off KS_TAIL_SRC=off KS_PAUSE=off"
( ./abl.sh P0_oval "KS_MASK=oval $OFF" ) &
( ABL_EXTRA="--mask tight --mask-chin 1 --mask-tsmooth 2 --neck-follow 0.15 --mask-chin-down 0.06" ./abl.sh P1_tight "KS_MASK=oval $OFF" ) &
( ./abl.sh P2_parse "KS_MASK=tight KS_CM_SHADOW= $OFF" ) &
( ./abl.sh P3_cmshadow "KS_MASK=tight $OFF" ) &
wait
( ./abl.sh P4_occl "KS_MASK=tight KS_CM_SHADOW= KS_OCCL=on KS_TAIL_SRC=off KS_PAUSE=off" ) &
( ./abl.sh P5_cap "KS_MASK=tight KS_CM_SHADOW= KS_OCCL=on KS_TAIL_SRC=off KS_PAUSE=off KS_CAP=1.0" ) &
( ./abl.sh P6_pause "KS_MASK=tight KS_CM_SHADOW= KS_OCCL=on KS_TAIL_SRC=off KS_CAP=1.0 KS_PAUSE=warp" ) &
wait; echo ALL_DONE >> done.txt
