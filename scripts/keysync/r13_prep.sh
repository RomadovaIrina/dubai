#!/bin/bash
# r13 KS inputs per video: the KeySync crop (ks_prep_src.sh; m03 = the speaker-only box KS_BOX_K 1.25 / KS_BOX_M 0.4), the source landmarks for the lag metric;
# after that video's E2E: the KeySync audio = the dub voice
cd /workspace/dub/keysync; M=/workspace/dub/inputs/mamakris; mkdir -p lm data/videos data/audios
for s in "qo3 /workspace/dub/inputs/test_videos_raw/03.mp4" "q18 $M/story-1774250817106-616905094.mp4" "m03 $M/story-1774249919482-367803614.MOV"; do set -- $s
  until [ -s $2 ] && [ $(ls $M | wc -l) -ge 9 -o "$1" = qo3 ]; do sleep 30; done
  if [ $1 = m03 ]; then export KS_BOX_K=1.25 KS_BOX_M=0.4; else unset KS_BOX_K KS_BOX_M; fi
  bash ks_prep_src.sh $1 $2 2>&1 | tail -2
  mkdir -p final/$1; ( cd /workspace/dub/dubai && /venv/mediapipe/bin/python scripts/quality/mp_landmarks.py --video $2 --out /workspace/dub/keysync/final/$1/src.mediapipe.npz | tail -1 )
  echo "PREP $1 $(cat data/box_$1.json)"
done
for i in qo3 q18 m03; do A=/workspace/dub/artifacts/r4a/r9_${i}_htd/$i
  until [ -s $A/dubbed_16k.wav ] && [ -s $A/final_mix_aac.m4a ]; do grep -q "E2E $i rc" r13_e2e.log && break; sleep 30; done
  [ -s $A/dubbed_16k.wav ] && ffmpeg -v error -y -i $A/dubbed_16k.wav -ac 1 -ar 16000 data/audios/$i.wav && echo "DUBAUDIO $i" || echo "NO_DUB $i"
done; echo R13_PREP_DONE
