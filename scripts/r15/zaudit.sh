#!/bin/bash
# r15 Z: the face-path code audit (coordination/audit_r15_face.md), generator-side items, as arms of the q2 teacher (patch_audit.py applied):
#   q2v2   production q2 + the default fixes (CF eye centroids without iris; failed rigid fits -> landmark similarity): the new baseline
#   q2cfa  + CF_ALIGN_FIT=lsq CF_MOUTH=rigid (CodeFormer alignment: least squares, mouth corners follow the head, not the RU mouth)
#   q2nb   + AF_UPPER=nobrow (rigid head motion without brows / forehead)
#   q2nr   + LS_NOSE_RIGID=1 (LatentSync nose anchor without the nostril wings)
#   q2ss0  + --steps-silent 0 (10 steps everywhere: no 10 / 20-step switch at window seams)
#   q2hy   + LS_SIL_HYST=1 (the switch with a majority-of-3 hysteresis)
#   q2vm   + LS_VAE_MEAN=1 (VAE posterior mean for the masked / reference latents)
#   q2g15  + --guidance 1.5 (production 2.0; the CFG lever of the worm task)
#   q2all  cfa + nb + nr + vm
# each + agent F's colour lock (<arm>lk), F's lip_flicker, SyncNet (alone and lk).    bash zaudit.sh [ids]    ARMS="..." to choose
cd /workspace/dub/dubai; . scripts/env.sh >/dev/null; X=/workspace/dub/r15w/eval
for id in ${*:-m18}; do
  for arm in ${ARMS:-q2v2 q2cfa q2nb q2nr q2ss0 q2hy q2vm q2g15 q2all}; do
    [ -s $X/${arm}_$id/${arm}_$id.mp4 ] && { echo "== $arm $id exists"; continue; }
    echo "== $id $arm $(date +%T)"
    case $arm in
      q2v2)  V=q2v2 bash scripts/r15/arm.sh $id;;
      q2cfa) CF_ALIGN_FIT=lsq CF_MOUTH=rigid V=q2cfa bash scripts/r15/arm.sh $id;;
      q2nb)  AF_UPPER=nobrow V=q2nb bash scripts/r15/arm.sh $id;;
      q2nr)  LS_NOSE_RIGID=1 V=q2nr bash scripts/r15/arm.sh $id;;
      q2ss0) GEN_EXTRA="--steps-silent 0" V=q2ss0 bash scripts/r15/arm.sh $id;;
      q2hy)  LS_SIL_HYST=1 V=q2hy bash scripts/r15/arm.sh $id;;
      q2vm)  LS_VAE_MEAN=1 V=q2vm bash scripts/r15/arm.sh $id;;
      q2g15) GEN_EXTRA="--guidance 1.5" V=q2g15 bash scripts/r15/arm.sh $id;;
      q2all) CF_ALIGN_FIT=lsq CF_MOUTH=rigid AF_UPPER=nobrow LS_NOSE_RIGID=1 LS_VAE_MEAN=1 V=q2all bash scripts/r15/arm.sh $id;;
    esac 2>&1 | grep "^=="
    [ -s $X/${arm}_$id/${arm}_$id.mp4 ] || continue
    SRC=$arm V=${arm}lk LT_ARGS="--mode color" LT_LABEL=c bash scripts/r15/comp_variant.sh $id 2>&1 | grep "^==" | tail -1
    for a in $arm ${arm}lk; do [ -s $X/${a}_$id/${a}_$id.mp4 ] || continue; bash scripts/r15/zlf.sh $id $a $arm
      flock /workspace/dub/gpu.lock python scripts/r11/sync_speech.py --video $X/${a}_$id/${a}_$id.mp4 \
        --audio /workspace/dub/artifacts/r4a/r9_q2_$id/q2_$id/dubbed_16k.wav --label ${a}_$id > $X/${a}_$id.sync.txt 2>&1; grep -h "== sync" $X/${a}_$id.sync.txt; done
  done
  echo "== $id AUDIT done $(date +%T)"
done
echo ZAUDIT_DONE
