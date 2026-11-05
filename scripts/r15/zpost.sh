#!/bin/bash
# r15 Z: the post steps of an arm whose generator finished (colour-lock composite <arm>lk, SyncNet, F lip_flicker, F explode), each skipped when done.
#   bash scripts/r15/zpost.sh <id> <arm> ...
cd /workspace/dub/dubai; . scripts/env.sh >/dev/null; X=/workspace/dub/r15w/eval; id=$1; shift
for arm in "$@"; do
  [ -s $X/${arm}_$id/${arm}_$id.mp4 ] || { echo "== zpost: no ${arm}_$id"; continue; }
  [ -s $X/${arm}lk_$id/${arm}lk_$id.mp4 ] || SRC=$arm V=${arm}lk LT_ARGS="--mode color" LT_LABEL=c bash scripts/r15/comp_variant.sh $id 2>&1 | grep "^==" | tail -1
  for a in $arm ${arm}lk; do [ -s $X/${a}_$id/${a}_$id.mp4 ] || continue
    grep -q "== sync" $X/${a}_$id.sync.txt 2>/dev/null || flock /workspace/dub/gpu.lock python scripts/r11/sync_speech.py --video $X/${a}_$id/${a}_$id.mp4 \
      --audio /workspace/dub/artifacts/r4a/r9_q2_$id/q2_$id/dubbed_16k.wav --label ${a}_$id > $X/${a}_$id.sync.txt 2>&1; grep -h "== sync" $X/${a}_$id.sync.txt
    grep -q "\"${a}_$id\"" $X/lipflick.jsonl 2>/dev/null || bash scripts/r15/zlf.sh $id $a $arm
    grep -q "\"${a}_$id\"" $X/explode.jsonl 2>/dev/null || bash scripts/r15/zexp.sh $id $a $arm 2>&1 | tail -1
  done
done
echo "ZPOST_DONE $id $*"
