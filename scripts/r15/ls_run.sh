#!/bin/bash
# r15 K: the LatentSync face GENERATOR stage alone (scripts/quality/candidate_video_stage.py with the r4_video.sh default generator args), the r6 student
# (R13_STUDENT_LORA, 4 steps, no CFG, no DeepCache) through scripts/r15/ls_trt_gen.py, on the q2 dub audio of r15-w (artifacts/r4a/r9_q2_<id>_htd).
#   bash scripts/r15/ls_run.sh <arm> <m03|old03>       env: LS_TRT_CAPTURE / LS_TRT_DIR / LS_TRT_FREE pass through; STEPS (4)
# -> /workspace/dub/lstrt/<arm>/<id>/: <id>_<arm>_25fps.mp4, the generator manifest (stage wall times, peak VRAM), gen.log ([lstrt] UNet totals), nvsmi.csv
set -u; A=$1; id=$2; R=/workspace/dub/dubai-ls; cd $R; export DUB_ROOT=$R; . scripts/env.sh >/dev/null
case $id in m03) SRC=/workspace/dub/inputs/mamakris/story-1774249919482-367803614.MOV;; old03) SRC=/workspace/dub/inputs/test_videos_raw/03.mp4;; m18) SRC=/workspace/dub/inputs/mamakris/story-1774250817106-616905094.mp4;; esac
AD=/workspace/dub/artifacts/r4a/r9_q2_${id}_htd/q2_$id; B=/workspace/dub/lstrt; W=$B/$A/$id; M=$B/manifest_$id.json; mkdir -p $W $B/base_$id
[ -s $M ] || python3 -c "import json; m=json.load(open('$AD/q2_${id}_r4b.manifest.json')); m['input']='$SRC'; m['work_dir']='$B/base_$id'; json.dump(m, open('$M', 'w'))"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True R13_STUDENT_LORA=${LORA:-/workspace/dub/r15k/r6_step_00250_ema.pt} R13_STUDENT_STEPS=${STEPS:-4} LS_TRT_TIMING=1
nvidia-smi --query-gpu=timestamp,memory.used,utilization.gpu --format=csv,noheader,nounits -lms 500 > $W/nvsmi.csv 2>/dev/null & NV=$!
t0=$(date +%s.%N)
flock /workspace/dub/gpu.lock python $R/scripts/r15/ls_trt_gen.py --id $id --tag ${A}_25fps --manifest $M --dubbed-dir $AD --out $W --work-out $W/video \
  --mux final_mix --gate-merge-gap-s 1.2 --mouth-only-pasteback on --mask-smooth 3 --ls-landmark-smooth 2 --codeformer-scope mouth --codeformer off \
  --face-stack mediapipe --dump-ls-raw on --anchor-filter flow:4 --anchor-round off --affine-precision fp32 --mask-track head --master-timing-mode fps \
  --guidance 2.0 --bridge-len 8 --bridge-steps 4 --max-ls-frames 400 > $W/gen.log 2>&1; rc=$?
kill $NV 2>/dev/null; t1=$(date +%s.%N)
echo "LS_RUN $A $id rc $rc wall $(python3 -c "print(round($t1 - $t0, 1))") s; $(grep -a "\[lstrt\] unet calls" $W/gen.log | tail -1); max nvidia-smi MiB $(cut -d, -f2 $W/nvsmi.csv | sort -n | tail -1)"
