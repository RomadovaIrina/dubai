#!/bin/bash
for v in m03 m18 old03 q02; do bash /workspace/dub/r16t/runjob.sh /workspace/dub/r16t/golden/gold_$v.json; done; touch /workspace/dub/r16t/golden/DONE
