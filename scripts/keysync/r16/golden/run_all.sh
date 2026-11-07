#!/bin/bash
# the golden runs, one after the other (each takes the GPU lock only for its GPU sections)
for v in m03 m18 old03 q02; do bash /workspace/dub/r16t/runjob.sh /workspace/dub/r16t/golden/gold_$v.json; done; echo GOLD_ALL_DONE > /workspace/dub/r16t/golden/DONE
