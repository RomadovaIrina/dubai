#!/bin/bash
G=/workspace/dub/r16t/golden; while [ ! -f $G/SHOTS_DONE ]; do sleep 20; done
bash /workspace/dub/r16t/runjob.sh $G/full_m03.json; touch $G/FULL_DONE
