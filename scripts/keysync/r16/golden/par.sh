#!/bin/bash
for v in m03 m18; do bash /workspace/dub/r16t/runjob.sh /workspace/dub/r16t/golden/par_$v.json; done; touch /workspace/dub/r16t/golden/PAR_DONE
