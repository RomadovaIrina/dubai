# r4 generator sweep driver: arms x windows (ls_window_ab.py) + r4_sweep_metrics.py; ARMS / IDS overridable
cd /workspace/dub/dubai; export DUB_ROOT=/workspace/dub/dubai; . scripts/env.sh >/dev/null; export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True OMP_NUM_THREADS=8
S=/workspace/dub/artifacts/r4sweep; AD=/workspace/dub/artifacts/r4a/cand2; O=$S/win; mkdir -p $O
COMMON="--mode ls --anchor-filter flow:4 --anchor-round off --affine-precision fp32 --mask-track head --mask-smooth 3 --ls-landmark-smooth 2 --codeformer off --face-stack mediapipe"
declare -A ARM=( [A0_base]="--guidance 2.0 --steps 20 --bridge-len 8 --bridge-steps 4 --seed 1247" [A1_g15]="--guidance 1.5 --steps 20 --bridge-len 8 --bridge-steps 4 --seed 1247"
  [A2_g25]="--guidance 2.5 --steps 20 --bridge-len 8 --bridge-steps 4 --seed 1247" [A3_s30]="--guidance 2.0 --steps 30 --bridge-len 8 --bridge-steps 4 --seed 1247"
  [A4_nobr]="--guidance 2.0 --steps 20 --seed 1247" [A5_seed7]="--guidance 2.0 --steps 20 --bridge-len 8 --bridge-steps 4 --seed 7"
  [A6_seed99]="--guidance 2.0 --steps 20 --bridge-len 8 --bridge-steps 4 --seed 99" )
for arm in ${ARMS:-A0_base A5_seed7 A6_seed99 A1_g15 A2_g25 A4_nobr A3_s30}; do
  for id in ${IDS:-04 05 03}; do
    W=$(readlink -f $S/work_$id); M=$S/${id}_base_25fps.manifest.json
    python3 -c "import json;[print(w['t0'],w['t1']) for w in json.load(open('$S/wins_$id.json'))['windows']]" | while read t0 t1; do
      tag=${arm}_${t0%.*}; grep -qs "check PASS" $O/${id}_$tag.log && continue
      python scripts/quality/ls_window_ab.py --manifest $M --t0 $t0 --t1 $t1 --tag $tag --out $O --rigid $W/master_rigid.npz $COMMON ${ARM[$arm]} < /dev/null > $O/${id}_$tag.log 2>&1 || { echo "FAIL $id $tag"; continue; }
      echo "== $id $tag $(date +%T)"
    done
  done
done
wait; echo SWEEP_DONE
