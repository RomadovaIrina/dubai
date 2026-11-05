#!/bin/bash
# one-shot: (re)start the m03 flash follower and the CFG-rescale queue (a script file, so pkill never matches the ssh command)
cd /workspace/dub/r15w
pkill -f "scripts/r15/zflash_follow.sh"
touch skip/q2ct5_m03
nohup bash /workspace/dub/dubai/scripts/r15/zflash_follow.sh m03 q2d q2g15 q2g125 q2ct7 q2rs7 q2rs10 > zrun_flash_m03.log 2>&1 &
ARMS="q2rs7 q2rs10" nohup bash /workspace/dub/dubai/scripts/r15/zcfg2.sh m18 m03 > zrun_cfg2.log 2>&1 &
sleep 2; cat zrun_cfg2.log; ps -eo pid,args | grep -E "zflash_[f]ollow|zcf[g]" | cut -c1-110
