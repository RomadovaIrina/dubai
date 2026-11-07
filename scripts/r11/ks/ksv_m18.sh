#!/bin/bash
while ! grep -q "ATTR_DONE ak" /workspace/dub/r11/attr_ks_m18.log; do sleep 20; done
cd /workspace/dub/dubai; . scripts/env.sh >/dev/null; B=/workspace/dub/artifacts/r4m
python scripts/r10/face_metrics.py --video $B/r9f_m18/m18_r9f_m18.mp4 --src /workspace/dub/r10/runs/m18/work/in/m18.mp4 --lm-src $B/r9g_m18/timing/m18_source_lm_F.npz --out /workspace/dub/r11/ksv/m18_r9f.json 2>&1 | grep "^==" | cut -c1-400
bash /workspace/dub/r11/ksv.sh m18 ksg_m18 ls_srt1 kv "fb|--fallback @FB" "nf|" "nfc|--colour-mouth-flat 0.6" "nfcm|--colour-mouth-flat 0.6 --mouth-region 0.25" "nfcw|--colour-mouth-flat 0.6 --chin-warp on" "nfcmw|--colour-mouth-flat 0.6 --mouth-region 0.25 --chin-warp on" "nfcn|--colour-mouth-flat 0.6 --retime nearest"
