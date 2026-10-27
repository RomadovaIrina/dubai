#!/bin/bash
# r6 G (generator ceiling): arms x the 9 worst lip-sync windows (3 per video: 03 / 04 / 05, worst 8-s dub-only SyncNet of the m3 run).
# Generator only (ls_window_ab.py: same master frames, dub audio, speech gate and mouth-only composite as m3; one setting changed per arm),
# then r4_sweep_metrics.py (SyncNet on dub speech, lip_r, flash, jaw) + r6_win_extra.py (twitch, chin wave, mouth/dub mismatch).
#   bash scripts/r6/r6_gen_sweep.sh            ARMS / IDS overridable (env)
set -u
cd /workspace/dub/dubai; export DUB_ROOT=/workspace/dub/dubai; . scripts/env.sh >/dev/null; export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True OMP_NUM_THREADS=8
S=/workspace/dub/artifacts/r6g; M3=/workspace/dub/artifacts/r4m/m3; O=$S/win; mkdir -p $O
COMMON="--mode ls --anchor-filter flow:4 --anchor-round off --affine-precision fp32 --mask-track head --mask-smooth 3 --ls-landmark-smooth 2 --codeformer off --face-stack mediapipe"
BR="--bridge-len 8 --bridge-steps 4"
declare -A ARM=( [G0_base]="--guidance 2.0 --steps 20 $BR" [G1_g25]="--guidance 2.5 --steps 20 $BR"
  [G2_g25ds05]="--guidance 2.5 --steps 20 $BR|dsmooth:0.5" [G3_g25ds10]="--guidance 2.5 --steps 20 $BR|dsmooth:1.0"
  [G4_g25rs07]="--guidance 2.5 --steps 20 $BR|rescale:0.7" [G5_g25dc15]="--guidance 2.5 --steps 20 $BR|dclip:1.5"
  [G6_g25int]="--guidance 2.5 --steps 20 $BR|interval:0-0.5:2.0" [G7_nobr]="--guidance 2.0 --steps 20"
  [G8_g25nobr]="--guidance 2.5 --steps 20" [G9_s12nobr]="--guidance 2.0 --steps 12" [G10_seed7]="--guidance 2.0 --steps 20 $BR --seed 7"
  [G11_g30ds10]="--guidance 3.0 --steps 20 $BR|dsmooth:1.0" [G12_g25s30]="--guidance 2.5 --steps 30 $BR" [G13_g225]="--guidance 2.25 --steps 20 $BR"
  [G14_g25ds10nobr]="--guidance 2.5 --steps 20|dsmooth:1.0" [G15_g25seed7]="--guidance 2.5 --steps 20 $BR --seed 7" )
for id in ${IDS:-04 05 03}; do   # windows of each video, once its m3 run exists
  W=$(ls -d $M3/work_${id}_m3 2>/dev/null) || continue; [ -f $S/wins_$id.json ] && continue
  python scripts/quality/r4_sweep_windows.py --video $W/${id}_m3_25fps.mp4 --dub /workspace/dub/artifacts/r4a/m3a/$id/dubbed_16k.wav --out $S/wins_$id.json --k 3 --w 8 > /dev/null 2>&1 || echo "FAIL wins $id"
done
for arm in ${ARMS:-G0_base G1_g25 G3_g25ds10 G2_g25ds05 G4_g25rs07 G7_nobr G10_seed7 MT_musetalk G5_g25dc15 G6_g25int G8_g25nobr G9_s12nobr G11_g30ds10 G13_g225 G14_g25ds10nobr G15_g25seed7}; do
  for id in ${IDS:-04 05 03}; do
    [ -f $S/wins_$id.json ] || continue; W=$M3/work_${id}_m3; MAN=$W/${id}_m3_25fps.manifest.json; RIG=$(ls $W/video/master_rigid.npz $W/*/master_rigid.npz 2>/dev/null | head -1)
    python3 -c "import json;[print(w['t0'],w['t1']) for w in json.load(open('$S/wins_$id.json'))['windows']]" | while read t0 t1; do
      tag=${arm}_${t0%.*}; grep -qs "check PASS" $O/${id}_$tag.log && continue; ts=$(date +%s)
      if [ $arm = MT_musetalk ]; then   # MuseTalk 1.5 generated outside, served through --precomputed (same gate / mask / composite)
        drv=$(ls $O/work_*_G0_base_${t0%.*}/drive_16k.wav 2>/dev/null | grep "/work_${id}_" | head -1); [ -n "$drv" ] || { echo "no G0 drive $id $t0"; continue; }
        MST=$(ls $W/video/master_25fps.mp4 $W/*/master_25fps.mp4 2>/dev/null | head -1)
        flock /workspace/dub/gpu.lock python scripts/r6/musetalk_window.py --master $MST --t0 $t0 --t1 $t1 --audio $drv --out $O/mt/${id}_${t0%.*}.mp4 > $O/${id}_${tag}_mt.log 2>&1 || { echo "FAIL mt $id $t0"; continue; }
        GEN="--precomputed $O/mt/${id}_${t0%.*}.mp4 --precomputed-label musetalk_v15"; CFG=""
      else
        A=${ARM[$arm]}; GEN="${A%%|*} --seed-default"; CFG=""; [[ "$A" == *"|"* ]] && CFG=${A#*|}
        GEN=${GEN/ --seed-default/}; [[ "$GEN" == *"--seed"* ]] || GEN="$GEN --seed 1247"
      fi
      DUB_LS_CFG="$CFG" flock /workspace/dub/gpu.lock python scripts/quality/ls_window_ab.py --manifest $MAN --t0 $t0 --t1 $t1 --tag $tag --out $O ${RIG:+--rigid $RIG} $COMMON $GEN < /dev/null > $O/${id}_$tag.log 2>&1 \
        || { echo "FAIL $id $tag"; continue; }
      echo "== $id $tag $(( $(date +%s) - ts )) s $(date +%T)"
    done
  done
done
echo R6_SWEEP_DONE
