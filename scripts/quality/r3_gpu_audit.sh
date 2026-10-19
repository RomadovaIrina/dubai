#!/bin/bash
# Round 3 D: all-GPU audit of the full E2E runner (run_clean_pipeline_05.py, clean0 video flags, r2 audio policy) on one video.
#   bash scripts/quality/r3_gpu_audit.sh <variant> <id...>
#   variants: gpu      = every model on the GPU (Silero cuda, Qwen CUDA llama.cpp n_gpu_layers=-1), per-stage load / unload (runner default)
#             cpuqwen  = same but Qwen on the CPU (the frozen baseline) and Silero on the CPU
#             resident = run_worker.py, ONE process for all <id...>, DUB_RESIDENT=1 with EVERY model kept resident (+ --qwen-resident on)
# Per run: stage JSONL (DUB_STAGE_LOG: wall clock, model loads, torch CUDA peaks) + nvidia-smi sampled every 200 ms -> r3_gpu_audit.py.
set -u
DUB_ROOT=/workspace/dub/dubai; export DUB_ROOT; . $DUB_ROOT/scripts/env.sh >/dev/null; cd $DUB_ROOT
var=$1; shift; R=/workspace/dub/artifacts/r3d/audit/$var; mkdir -p $R
FLAGS="--target-lang en --alignment burst --video-backend optimized --codeformer off --face-stack mediapipe --audio-policy r2 \
  --gate-merge-gap-s 1.2 --mouth-only-pasteback on --codeformer-scope mouth --mask-smooth 3 --ls-landmark-smooth 2"
LLAMA_CUDA=$DUB_ROOT/third_party/llama_cuda
case $var in
  gpu|resident) GF="--qwen-gpu-layers -1 --vad-device cuda"; export PYTHONPATH=$LLAMA_CUDA${PYTHONPATH:+:$PYTHONPATH};;
  cpuqwen) GF="";;
  *) echo "unknown variant $var"; exit 2;;
esac
run_one() {   # $1 = log tag, rest = command
  tag=$1; shift; export DUB_STAGE_LOG=$R/${tag}.stages.jsonl; : > $DUB_STAGE_LOG
  nvidia-smi --query-gpu=timestamp,memory.used,utilization.gpu --format=csv,noheader,nounits -lms 200 > $R/${tag}.nvsmi.csv 2>/dev/null & NV=$!
  s=$(date +%s); flock /workspace/dub/gpu.lock "$@" > $R/${tag}.log 2>&1; rc=$?
  kill $NV 2>/dev/null; wait $NV 2>/dev/null
  echo "== audit $var $tag rc=$rc wall $(( $(date +%s) - s )) s"
}
if [ $var = resident ]; then
  export DUB_RESIDENT_KEYS="silero_vad,whisper_asr,pyannote,chatterbox_v3,tts_qa,latentsync,router"
  run_one worker python scripts/pilot/run_worker.py --run-root $R --ids "$@" -- $FLAGS $GF --qwen-resident on
else
  for id in "$@"; do
    in=$(ls /workspace/dub/inputs/test_videos_raw/$id.* | head -1)
    export DUB_STAGE_JOB=$id; run_one $id python scripts/pilot/run_clean_pipeline_05.py --input $in --output $R/${id}_final.mp4 --work-dir $R/work $FLAGS $GF
  done
fi
python scripts/quality/r3_gpu_audit.py --dir $R > $R/audit.md; cat $R/audit.md | head -60
