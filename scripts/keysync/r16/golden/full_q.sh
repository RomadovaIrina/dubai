#!/bin/bash
for j in full_old03 full_m18 long5; do bash /workspace/dub/r16t/runjob.sh /workspace/dub/r16t/golden/$j.json; touch /workspace/dub/r16t/golden/$j.done; done
