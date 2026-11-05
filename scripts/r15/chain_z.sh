#!/bin/bash
# r15 Z (worms): generator-side and input-crop arms of the q2 teacher on the q2 baseline dubs (m03 with face units, default), each also with
# agent F's lip colour lock (r7_m7_comp LT_ARGS="--mode color", a composite-only rerun: <arm>lk), then SyncNet / F's lip_flicker / clips.
#   q2d   production q2 + F3 + LS_DUMP_DIR (the crop-space stages b / c of the decomposition)
#   q2as  q2 + LS_AFFINE_SMOOTH=9 (zero-lag SG smoothing of LatentSync's 3-point crop affine; the same transform pastes back)
#   q2c8  the teacher with c8 (LS_STRIDE=8 continue: the reference for the continue-path arms)
#   te2   q2c8 + LS_ENS=2 (2-noise latent ensemble, per-step averaged x0)
#   te2k  te2 + LS_KF=2 (keyframes / 2 + DIS interpolation: ~ production cost)
#   q2s4  the teacher with LS_STRIDE=4 continue (more overlap)
#   bash scripts/r15/chain_z.sh [m18 old03 m03]       ARMS="..." to choose
set -u
cd /workspace/dub/dubai; X=/workspace/dub/r15w/eval
for id in ${*:-m18 old03 m03}; do
  for arm in ${ARMS:-q2d q2as q2c8 te2 te2k q2s4}; do
    echo "== $id $arm $(date +%T)"
    case $arm in
      q2d)  LS_DUMP_DIR=/workspace/dub/r15w/zdump/$id V=q2d bash scripts/r15/arm.sh $id;;
      q2as) LS_AFFINE_SMOOTH=9 V=q2as bash scripts/r15/arm.sh $id;;
      q2c8) LS_STRIDE=8 LS_OVERLAP=continue V=q2c8 bash scripts/r15/arm.sh $id;;
      te2)  LS_STRIDE=8 LS_OVERLAP=continue LS_ENS=2 V=te2 bash scripts/r15/arm.sh $id;;
      te2k) LS_STRIDE=8 LS_OVERLAP=continue LS_ENS=2 LS_KF=2 V=te2k bash scripts/r15/arm.sh $id;;
      q2s4) LS_STRIDE=4 LS_OVERLAP=continue V=q2s4 bash scripts/r15/arm.sh $id;;
    esac 2>&1 | grep "^=="
    [ -s $X/${arm}_$id/${arm}_$id.mp4 ] || continue
    SRC=$arm V=${arm}lk LT_ARGS="--mode color" LT_LABEL=c bash scripts/r15/comp_variant.sh $id 2>&1 | grep "^==" | tail -1   # + the colour lock
    for a in $arm ${arm}lk; do [ -s $X/${a}_$id/${a}_$id.mp4 ] && bash scripts/r15/zlf.sh $id $a $arm; done
    [ -s $X/${arm}_$id.sync.txt ] || flock /workspace/dub/gpu.lock python scripts/r11/sync_speech.py --video $X/${arm}_$id/${arm}_$id.mp4 \
      --audio /workspace/dub/artifacts/r4a/r9_q2_$id/q2_$id/dubbed_16k.wav --label ${arm}_$id > $X/${arm}_$id.sync.txt 2>&1; grep -h "== sync" $X/${arm}_$id.sync.txt
  done
  echo "== $id Z done $(date +%T)"
done
echo Z_CHAIN_DONE
