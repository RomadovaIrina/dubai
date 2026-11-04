cd /workspace/dub/dubai; B=/workspace/dub/artifacts/r4m
bash scripts/r12/abl.sh old03 "fcall=--colour-match all" "fcall45=--colour-match all --mouth-region 0.45" "fmr45=--mouth-region 0.45"
. scripts/env.sh >/dev/null
for t in amr38 fcall fcall45 fmr45 a0; do flock /workspace/dub/gpu.lock python scripts/r11/sync_speech.py --video $B/r9f_old03$t/old03_r9f_old03$t.mp4 --audio /workspace/dub/artifacts/r4a/r9_old03_htd/old03/dubbed_16k.wav --label "old03 $t"; done
echo FIX1_DONE
