#!/bin/bash
# r11 FACE: ks_fix on m18 too, then composites (no fallback + colour-mouth-flat 0.6) on the repaired layers of m18 + old03 -> SyncNet / ring / chin swim
cd /workspace/dub/dubai; . scripts/env.sh >/dev/null; B=/workspace/dub/artifacts/r4m
W=$B/ksg_m18/work_m18_ksg_m18; P=$W/m18_ksg_m18_25fps
C="--layer ${P}_ls_srt1.mp4 --master $W/video/master_25fps.mp4 --box /workspace/dub/ksx/keysync/data/box_q18.json --layer-lm ${P}_ls_srt1.mediapipe.npz --src-lm $B/ksg_m18/timing/m18_source_lm_F.npz"
python scripts/r11/ks_fix.py $C --out ${P}_ls_kfglt.mp4 --guard 3 --skin-cm 12 --lip-chroma 1 --chin tex &
python scripts/r11/ks_fix.py $C --out ${P}_ls_kfgls.mp4 --guard 3 --skin-cm 12 --lip-chroma 1 --chin src &
wait
python scripts/r11/sync_speech.py --video $B/r9f_qo3/qo3_r9f_qo3.mp4 --audio /workspace/dub/artifacts/r4a/r9_qo3_htd/qo3/dubbed_16k.wav --label "qo3 r9+fix"
cd /workspace/dub/r11
bash ksv.sh m18 ksg_m18 ls_kfglt kt "nfc|--colour-mouth-flat 0.6" &
bash ksv.sh m18 ksg_m18 ls_kfgls ks "nfc|--colour-mouth-flat 0.6" &
bash ksv.sh qo3 ksga_qo3 ls_srt1 ko "nfc|--colour-mouth-flat 0.6" &
bash ksv.sh qo3 ksga_qo3 ls_kfglt kot "nfc|--colour-mouth-flat 0.6" &
bash ksv.sh qo3 ksga_qo3 ls_kfgls kos "nfc|--colour-mouth-flat 0.6" &
wait; echo KSFIX2_DONE
