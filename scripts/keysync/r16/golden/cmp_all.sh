#!/bin/bash
G=/workspace/dub/r16t/golden; R=/workspace/dub/r16in; Q=$R/day1/HANDOFF_Q; RT=/workspace/dub/dubai/reports/experiments/r16-qa/routing; S=/workspace/dub/dubai/scripts/keysync/r16/gold_all.sh

bash $S m03 /workspace/dub/inputs/mamakris/story-1774249919482-367803614.MOV $R/golden/m03_keysync_clean5.mp4 $R/HANDOFF_K/paste_inputs/m03 $RT/m03.json > $G/cmp_m03.out 2>&1
bash $S m18 /workspace/dub/inputs/mamakris/story-1774250817106-616905094.mp4 $R/golden/m18_keysync_clean5.mp4 $R/HANDOFF_K/paste_inputs/m18 > $G/cmp_m18.out 2>&1
bash $S old03 /workspace/dub/inputs/test_videos_raw/03.mp4 $R/golden/old03_keysync_clean5.mp4 $R/HANDOFF_K/paste_inputs/old03 > $G/cmp_old03.out 2>&1
bash $S q02 $Q/clips/q02_vera.mp4 $R/golden/q02_keysync_r16_route_track.mp4 $Q/mixes/q02_vera $RT/q02_vera.json 40.88 > $G/cmp_q02.out 2>&1
touch $G/CMP_DONE
