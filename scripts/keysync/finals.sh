#!/bin/bash
cd /workspace/dub/keysync; M=/workspace/dub/inputs/mamakris
until grep -q E2E3_DONE /workspace/dub/r12k/e2e3.log; do sleep 20; done
bash ks_final.sh qo3 /workspace/dub/inputs/test_videos_raw/03.mp4 old03
bash ks_final.sh q18 $M/story-1774250817106-616905094.mp4 m18
bash ks_final.sh m03 $M/story-1774249919482-367803614.MOV m03
echo FINALS_DONE
