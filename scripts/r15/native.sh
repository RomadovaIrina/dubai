#!/bin/bash
# r15 W0: "TBDub native" = TBDub's OWN paste-back output with none of our post-chain (no CodeFormer, no mouth-only mask, no fallback / repair,
# no 25 -> 30 composite): the pre-CodeFormer raw dump (_ls_pre.mp4, 25 fps master; frames outside the speech gate = the master) of the
# whole-video TBDub arm (tbw), muxed with the same demucs mix -> eval/tbnat_<id>/tbnat_<id>.mp4, its SyncNet (--native), the eye clips vs q2.
#   bash scripts/r15/native.sh <old03|m03|m18> ...
set -u
cd /workspace/dub/dubai; export DUB_ROOT=/workspace/dub/dubai; . scripts/env.sh >/dev/null
X=/workspace/dub/r15w/eval; A=/workspace/dub/artifacts/r4a; B=/workspace/dub/artifacts/r4m
for id in "$@"; do
  ID=q2_$id; W=$B/r9g_${ID}_tbw/work_${ID}_r9g_${ID}_tbw; P=$W/${ID}_r9g_${ID}_tbw_25fps_ls_pre.mp4; O=$X/tbnat_$id; mkdir -p $O
  [ -s $P ] || { echo "== tbnat_$id FAIL: no $P"; continue; }
  ffmpeg -v error -y -i $P -i $A/r9_${ID}_htd/$ID/final_mix_aac.m4a -map 0:v:0 -map 1:a:0 -c:v libx264 -crf 16 -preset fast -pix_fmt yuv420p -c:a copy -shortest $O/tbnat_$id.mp4
  [ -s $X/tbnat_$id.sync.txt ] || flock /workspace/dub/gpu.lock python scripts/r11/sync_speech.py --video $O/tbnat_$id.mp4 --audio $A/r9_$ID/$ID/dubbed_16k.wav --native --label tbnat_$id 2>&1 | grep "^== sync" > $X/tbnat_$id.sync.txt
  cat $X/tbnat_$id.sync.txt
  bash scripts/r15/pair.sh $id q2 tbnat "q2 + F3 (production)" "TBDub native (own paste-back, 25 fps)" > /dev/null 2>&1
  echo "== tbnat_$id done"
done
