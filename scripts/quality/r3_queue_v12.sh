#!/bin/bash
# r3 D queue v12 (user defects 19.10): audio-r4 (phrase-cut fix + loudness) -> r3final = A R3 generator (+ B fps master) on audio-r4
#   -> FIX30 mode r3fix (B final composite + confidence-checked 25->30 retime) -> flash / hard-cut checks -> outbox
cd /workspace/dub/dubai; . scripts/env.sh >/dev/null; V=/workspace/dub/artifacts/r3d/video; A4=/workspace/dub/artifacts/r3d/audio-r4; T=/workspace/dub/artifacts/r3d/timing
until grep -q AUDIO_R4_DONE /workspace/dub/artifacts/r3d/audio_r4.log; do sleep 30; done
for id in 04 03 05 01 02; do
  [ -f $A4/$id/dubbed_16k.wav ] || { echo "WARN no audio-r4 $id"; continue; }
  DUBBED_DIR_TPL=$A4/$id MANIFEST_TPL=$A4/$id/${id}_r4.manifest.json bash scripts/quality/r3_video.sh r3final $id
  ( FIX30_MODE=r3fix FIX30_THREADS=8 bash scripts/quality/r3_fix30.sh $V/${id}_r3final.manifest.json
    SRC=$(ls /workspace/dub/inputs/test_videos_raw/$id.* | head -1); LM=$T/${id}_source_lm_F.npz; [ -f $LM ] && LMA="--lm $LM" || LMA=""
    python scripts/quality/r3_flash.py --video $V/${id}_r3final_fix30.mp4 --src $SRC $LMA --out $V/${id}_r3final_fix30.flash.json | tail -1 | cut -c1-200 ) &
done
wait; echo "R3FINAL_DONE $(date +%T)"
bash scripts/quality/r3_final_outbox.sh; echo "OUTBOX_DONE $(date +%T)"
