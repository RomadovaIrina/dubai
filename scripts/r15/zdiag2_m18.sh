#!/bin/bash
# r15 Z diagnostics 2 on m18: milder CFG than LS_CFG_TRUNC=3 (which cut the blinks ~70 % but collapsed SyncNet 8.1 -> 5.5).
# q2 runs 10 steps on speech (20 on silence; the trunc is scaled per group): ct5 = CFG in the first half, ct7 = the first 70 %; g125 = guidance 1.25
# everywhere (prod 1.5). Each + colour lock + F lip_flicker + SyncNet (alone and lk).
cd /workspace/dub/dubai; . scripts/env.sh >/dev/null; X=/workspace/dub/r15w/eval; id=m18
for arm in ${ARMS:-q2ct5 q2ct7 q2g125}; do
  echo "== $arm $(date +%T)"
  case $arm in
    q2ct5)  LS_CFG_TRUNC=5 V=q2ct5 bash scripts/r15/arm.sh $id;;
    q2ct7)  LS_CFG_TRUNC=7 V=q2ct7 bash scripts/r15/arm.sh $id;;
    q2g125) GEN_EXTRA="--guidance 1.25" V=q2g125 bash scripts/r15/arm.sh $id;;
  esac 2>&1 | grep "^=="
  [ -s $X/${arm}_$id/${arm}_$id.mp4 ] || continue
  SRC=$arm V=${arm}lk LT_ARGS="--mode color" LT_LABEL=c bash scripts/r15/comp_variant.sh $id 2>&1 | grep "^==" | tail -1
  for a in $arm ${arm}lk; do [ -s $X/${a}_$id/${a}_$id.mp4 ] || continue; bash scripts/r15/zlf.sh $id $a $arm
    flock /workspace/dub/gpu.lock python scripts/r11/sync_speech.py --video $X/${a}_$id/${a}_$id.mp4 \
      --audio /workspace/dub/artifacts/r4a/r9_q2_$id/q2_$id/dubbed_16k.wav --label ${a}_$id > $X/${a}_$id.sync.txt 2>&1; grep -h "== sync" $X/${a}_$id.sync.txt; done
done
echo DIAG2_DONE
