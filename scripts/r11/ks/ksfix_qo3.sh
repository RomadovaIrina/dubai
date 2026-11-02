#!/bin/bash
# r11 FACE: KeySync layer repair variants on old03 (ks_fix.py) + the lip / chin sheet on the fixed frames
cd /workspace/dub/dubai; . scripts/env.sh >/dev/null
B=/workspace/dub/artifacts/r4m; W=$B/ksga_qo3/work_qo3_ksga_qo3; L=$B/r9g_qo3/work_qo3_r9g_qo3; P=$W/qo3_ksga_qo3_25fps
C="--layer ${P}_ls_srt1.mp4 --master $W/video/master_25fps.mp4 --box /workspace/dub/ksx/keysync/data/box_qo3.json --layer-lm ${P}_ls_srt1.mediapipe.npz --src-lm $B/ksga_qo3/timing/qo3_source_lm_F.npz"
python scripts/r11/ks_fix.py $C --out ${P}_ls_kfg.mp4 --guard 3 --guard-report /workspace/dub/r11/ulip_qo3/guard.json &
python scripts/r11/ks_fix.py $C --out ${P}_ls_kfgl.mp4 --guard 3 --skin-cm 12 --lip-chroma 1 &
python scripts/r11/ks_fix.py $C --out ${P}_ls_kfglt.mp4 --guard 3 --skin-cm 12 --lip-chroma 1 --chin tex &
python scripts/r11/ks_fix.py $C --out ${P}_ls_kfgls.mp4 --guard 3 --skin-cm 12 --lip-chroma 1 --chin src &
wait
python scripts/r11/upper_lip.py --src $W/video/master_25fps.mp4 --src-lm $B/ksga_qo3/timing/qo3_source_lm_F.npz --ls $L/qo3_r9g_qo3_25fps_ls_raw.mp4 --ls-lm $L/qo3_r9g_qo3_25fps_ls_raw.mediapipe.npz \
  --ks-lm ${P}_ls_srt1.mediapipe.npz --rend ks=${P}_ls_srt1.mp4 --rend guard=${P}_ls_kfg.mp4 --rend lipfix=${P}_ls_kfgl.mp4 --rend chintex=${P}_ls_kfglt.mp4 --rend chinsrc=${P}_ls_kfgls.mp4 \
  --frames 224,440,790,856,894,1634,2022,2112,2194,2196,2208 --out /workspace/dub/r11/ulip_qo3 --tag step1 | grep -v "^\.\."
echo KSFIX_DONE
