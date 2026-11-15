#!/bin/bash
# r20 harness: KeySync fast (the e2e settings of ks_face.sh) on ready crops, several jobs in ONE process.
#   gen.sh <name> <jobs.txt (paths as seen in the container: /r18/...)> [K=V ...]     -> /root/r18/gen/<name>/<tag>/*.mp4, log /root/r18/gen/<name>.log
set -uo pipefail
B=/root/r18; CODE=$B/code; IMG=${R18_IMG:-ghcr.io/romadovairina/dubai:latest}; VOL=dabai-models
name=$1; jobs=$2; shift 2; EV=(); for kv in "$@"; do EV+=(-e "$kv"); done
mkdir -p $B/gen/$name $B/kit; rm -f $B/kit/*.py
for f in $CODE/scripts/keysync/*.py $CODE/scripts/keysync/r15/*.py $CODE/scripts/keysync/r16/*.py; do cp $f $B/kit/; done
docker run --rm --gpus all --shm-size 16g -v $VOL:/workspace/dub/dubai/models -v $CODE/scripts:/workspace/dub/dubai/scripts -v $B:/r18 -v $B/data:/data \
  -e KS_SEED=${KS_SEED:-1234} "${EV[@]}" $IMG bash -c '
  set -o pipefail; KS=${KS_ROOT:-/opt/keysync}; R=/workspace/dub/dubai
  ENG=$(KS_GPU_LOCK=/dev/null bash $R/scripts/keysync/r16/ks_engines.sh 2> /r18/gen/'$name'.eng.log | tail -1); [ -d "$ENG" ] || { tail -5 /r18/gen/'$name'.eng.log; exit 1; }
  [ -n "${KS_TORCH_HOME:-}" ] && export TORCH_HOME=$KS_TORCH_HOME
  GATE="${KS_GATE-thr=0.12,cap=0.25,hi=8,lo=2}"
  cd $KS/code && . ../venv/bin/activate && env KS_DC=2:0 KS_LMK_CACHE=${KS_LMK_CACHE:-/data/ks_lmk} ${GATE:+KS_GATE=$GATE} \
    python /r18/kit/ks_fast.py --jobs '$jobs' --out /r18/gen/'$name' --cfg _s8 --vae-half --compile unet --trt $ENG --interp-lora $KS_LORA --interp-steps ${KS_INTERP_STEPS:-2} --vae-batch 8 --vae-compile' > $B/gen/$name.log 2>&1
rc=$?; grep -E "^KS_JOB|^KS_LREF|^KS_GATE hi" $B/gen/$name.log | cut -c1-200; echo "GEN $name rc=$rc"
