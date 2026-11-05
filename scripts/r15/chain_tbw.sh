#!/bin/bash
# r15 W0b: TBDub whole-video mode (one generation over the master + the whole dub, sliced per segment) + OpenCV colour match
cd /workspace/dub/dubai; X=/workspace/dub/r15w/eval
for id in old03 m03 m18; do
  until [ -s $X/q2_$id/q2_$id.mp4 ] || { grep -q BASE_DONE /workspace/dub/r15w/base_q2b.log 2>/dev/null && [ "$id" != old03 ]; }; do sleep 20; done
  [ -s $X/q2_$id/q2_$id.mp4 ] || continue
  until [ -s $X/tb_$id/timing.json ] || grep -q FAIL $X/tb_$id/logs/face.log 2>/dev/null; do sleep 20; done   # after the per-segment TBDub arm (no two TBDub workers at once)
  echo "== $id tbw $(date +%T)"; V=tbw TBDUB_WHOLE=1 TBDUB_FASTCC=1 R10_GEN_PY=scripts/r15/tbdub_gen.py bash scripts/r15/arm.sh $id 2>&1 | grep "^=="
  bash scripts/r15/pair.sh $id tb tbw "TBDub per segment" "TBDub whole video" > /dev/null 2>&1
done
echo TBW_CHAIN_DONE
