#!/bin/bash
# r15 Z: milder CFG on a video (coordinator, m03 first: the mouth-shape SNAPS = the user's worst m03 complaint), every arm with CF_EYE_IRIS=1
# (= production CodeFormer eye points) so CFG is the only difference to q2d (production, + LS_DUMP_DIR); m03 runs with face units (arm.sh default).
#   q2d (prod) | q2g15 guidance 1.5 | q2g125 guidance 1.25 | q2ct7 CFG in the first 70 % of steps | q2ct5 the first 50 %   (production guidance 2.0)
# each + F's colour lock (<arm>lk), F lip_flicker, F explode (snaps), SyncNet.    bash zcfg.sh <id> ...    ARMS="..." to choose
cd /workspace/dub/dubai; . scripts/env.sh >/dev/null; X=/workspace/dub/r15w/eval; export CF_EYE_IRIS=1
for id in "$@"; do
  for arm in ${ARMS:-q2d q2g15 q2g125 q2ct7 q2ct5}; do
    echo "== $id $arm $(date +%T)"
    case $arm in
      q2d)    LS_DUMP_DIR=/workspace/dub/r15w/zdump/$id V=q2d bash scripts/r15/arm.sh $id;;
      q2g15)  GEN_EXTRA="--guidance 1.5" V=q2g15 bash scripts/r15/arm.sh $id;;
      q2g125) GEN_EXTRA="--guidance 1.25" V=q2g125 bash scripts/r15/arm.sh $id;;
      q2ct7)  LS_CFG_TRUNC=7 V=q2ct7 bash scripts/r15/arm.sh $id;;
      q2ct5)  LS_CFG_TRUNC=5 V=q2ct5 bash scripts/r15/arm.sh $id;;
    esac 2>&1 | grep "^=="
    [ -s $X/${arm}_$id/${arm}_$id.mp4 ] || continue
    [ -s $X/${arm}lk_$id/${arm}lk_$id.mp4 ] || SRC=$arm V=${arm}lk LT_ARGS="--mode color" LT_LABEL=c bash scripts/r15/comp_variant.sh $id 2>&1 | grep "^==" | tail -1
    for a in $arm ${arm}lk; do [ -s $X/${a}_$id/${a}_$id.mp4 ] || continue
      grep -q "== sync" $X/${a}_$id.sync.txt 2>/dev/null || flock /workspace/dub/gpu.lock python scripts/r11/sync_speech.py --video $X/${a}_$id/${a}_$id.mp4 \
        --audio /workspace/dub/artifacts/r4a/r9_q2_$id/q2_$id/dubbed_16k.wav --label ${a}_$id > $X/${a}_$id.sync.txt 2>&1; grep -h "== sync" $X/${a}_$id.sync.txt
      bash scripts/r15/zexp.sh $id $a $arm 2>&1 | tail -1; bash scripts/r15/zlf.sh $id $a $arm; done
  done
  echo "== $id CFG done $(date +%T)"
done
echo ZCFG_DONE
