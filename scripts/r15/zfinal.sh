#!/bin/bash
# r15 Z final pass: the missing SyncNet of every arm (+ lk) (chain_z.sh lacked the env for its own call), the table -> eval/ZTABLE.md, and
# the eye clips at the user's spots (0.5x): q2d | <arm> | <arm>lk 3-panels for the arms given (PANELS="te2 q2c8 ...").
#   bash scripts/r15/zfinal.sh [ids]
set -u
cd /workspace/dub/dubai; . scripts/env.sh >/dev/null; X=/workspace/dub/r15w/eval; A=/workspace/dub/artifacts/r4a
ARMS="q2d q2as q2c8 te2 te2k q2s4 q2c8rf q2ct3 s4c8"
for id in ${*:-m18 old03 m03}; do
  for arm in $ARMS; do for a in $arm ${arm}lk; do
    v=$X/${a}_$id/${a}_$id.mp4; [ -s $v ] || continue
    grep -q "== sync" $X/${a}_$id.sync.txt 2>/dev/null && continue
    flock /workspace/dub/gpu.lock python scripts/r11/sync_speech.py --video $v --audio $A/r9_q2_$id/q2_$id/dubbed_16k.wav --label ${a}_$id > $X/${a}_$id.sync.txt 2>&1
  done; done
done
python3 scripts/r15/ztable.py --ids ${*:-m18 old03 m03} --arms $ARMS > $X/ZTABLE.md; cat $X/ZTABLE.md
for id in ${*:-m18 old03 m03}; do for arm in ${PANELS:-}; do
  [ -s $X/${arm}_$id/${arm}_$id.mp4 ] && [ -s $X/${arm}lk_$id/${arm}lk_$id.mp4 ] && \
    bash scripts/r15/panel3.sh $id q2d $arm ${arm}lk "q2 + F3 (production)" "$arm" "$arm + colour lock" 2>&1 | tail -1
done; done
echo ZFINAL_DONE
