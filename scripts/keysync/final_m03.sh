#!/bin/bash
cd /workspace/dub/keysync; until grep -q FINALS2_DONE finals2.log; do sleep 15; done
rm -rf final/m03; bash finals2_m03.sh > finals2_m03.log 2>&1
