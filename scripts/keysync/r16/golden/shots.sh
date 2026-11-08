#!/bin/bash
G=/workspace/dub/r16t/golden; while [ ! -f $G/R0_DONE ]; do sleep 20; done
bash /workspace/dub/r16t/runjob.sh $G/shots_q02.json; touch $G/SHOTS_DONE
