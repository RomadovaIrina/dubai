#!/bin/bash
G=/workspace/dub/r16t/golden; while [ ! -f $G/PAR_DONE ]; do sleep 20; done
for v in m18 old03; do bash /workspace/dub/r16t/runjob.sh $G/r0_$v.json; done; touch $G/R0_DONE
