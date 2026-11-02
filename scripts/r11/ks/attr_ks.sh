#!/bin/bash
while ! grep -q ATTR_DONE /workspace/dub/r11/attr_ls_m18.log; do sleep 20; done
cd /workspace/dub/dubai; . scripts/env.sh >/dev/null; export DUB_ROOT=/workspace/dub/dubai DUB_ALLOW_NC=1
W=/workspace/dub/artifacts/r4m/ksg_m18/work_m18_ksg_m18; A=/workspace/dub/artifacts/r4a/r9_m18_htd/m18/dubbed_16k.wav
for L in ls_raw ls_srt1; do flock /workspace/dub/gpu.lock python scripts/r11/sync_speech.py --video $W/m18_ksg_m18_25fps_$L.mp4 --audio $A --native --label "KS layer $L"; done
bash /workspace/dub/r11/attr2.sh m18 ksg_m18 $W/m18_ksg_m18_25fps_ls_srt1.mp4 ak base nofb bare mr0 tex0 san0 near sh0 cm0 lg0 reg0
