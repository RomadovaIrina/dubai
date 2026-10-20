#!/bin/bash
# Round 4 AUDIO agent: ONE command, all five videos from the raw input, the same flags for every video (no per-video settings).
#   1. audio stage = v4's audio-r4 (r3_audio_r4.sh flags: r2 policy, speakable text, lexicon, cut snap, equal-power fades, Qwen on the GPU)
#   2. bed remix + static loudness (r3_remix.py; v4 = audio-r4b defaults)
#   3. new audio over the UNTOUCHED source video (<id>_over-original.mp4) + r4_audio_metrics.py
#     bash scripts/r4_audio.sh <tag> [ids]     -> /workspace/dub/artifacts/r4a/<tag>/{<id>/ (mix dir), runs/, <id>_over-original.mp4, <id>.metrics.json}
# env: AUDIO_EXTRA = extra run_clean_pipeline_05.py flags; REMIX_ARGS = r3_remix.py flags; REUSE=<tag> = take the TTS run of another tag (bed-only A/B);
#      PAR = videos in parallel on the GPU (default 2)
set -u
DUB_ROOT=/workspace/dub/dubai; export DUB_ROOT; . $DUB_ROOT/scripts/env.sh >/dev/null; cd $DUB_ROOT
TAG=$1; shift; IDS=${*:-01 02 03 04 05}; A=/workspace/dub/artifacts/r4a; O=$A/$TAG; mkdir -p $O/runs $O/mix_src $A/sep
export PYTHONPATH=third_party/llama_cuda${PYTHONPATH:+:$PYTHONPATH} PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True DUB_CUT_SNAP=1 DUB_EQPOWER_FADE=1
echo "r4_audio $TAG commit $(git rev-parse --short HEAD) AUDIO_EXTRA='${AUDIO_EXTRA:-}' REMIX_ARGS='${REMIX_ARGS:-}' REUSE='${REUSE:-}'" > $O/README.txt
one() {
  id=$1; src=$(ls /workspace/dub/inputs/test_videos_raw/$id.* | head -1); s=$(date +%s)
  if [ -n "${REUSE:-}" ]; then W=$(readlink -f $A/$REUSE/mix_src/$id); else
    python scripts/pilot/run_clean_pipeline_05.py --input $src --output $O/runs/${id}_r4a.mp4 --target-lang en --alignment burst \
      --stop-after audio --codeformer off --face-stack mediapipe --audio-policy r2 --qwen-gpu-layers -1 --tts-text-norm on --qa-whisper-compute int8_float16 \
      ${AUDIO_EXTRA:-} --work-dir $O/runs/$id > $O/runs/${id}.log 2>&1 || { echo "FAIL audio $id"; return; }
    W=$O/runs/$id/work_$(basename ${src%.*}); cp $O/runs/${id}_r4a.manifest.json $W/${id}_r4a.manifest.json
  fi
  ln -sfn $W $O/mix_src/$id; [ -d $A/sep/$id ] || { mkdir -p $A/sep/$id; cp $W/bg_44k.wav $W/vocals_44k.wav $A/sep/$id/; }
  python scripts/quality/r3_remix.py --src $O/mix_src --out $O ${REMIX_ARGS:-} $id > $O/runs/${id}.remix.log 2>&1 || { echo "FAIL remix $id"; return; }
  ffmpeg -loglevel error -y -i $src -i $O/$id/final_mix_aac.m4a -map 0:v:0 -map 1:a:0 -c copy -shortest $O/${id}_over-original.mp4
  python scripts/quality/r4_audio_metrics.py --mix $O/${id}_over-original.mp4 --src $src --sep $A/sep/$id --run $O/$id --manifest $O/$id/${id}_r4b.manifest.json --wer \
    --json $O/${id}.metrics.json > $O/runs/${id}.metrics.log 2>&1 || echo "WARN metrics $id"
  echo "== $id $TAG $(( $(date +%s) - s )) s"
}
nvidia-smi --query-gpu=timestamp,memory.used,utilization.gpu --format=csv,noheader,nounits -lms 1000 > $O/nvsmi.csv 2>/dev/null & NV=$!
i=0; for id in $IDS; do one $id & i=$((i+1)); [ $((i % ${PAR:-2})) -eq 0 ] && wait; done; wait
kill $NV 2>/dev/null; echo "peak VRAM MiB $(awk -F, '{if($2>m)m=$2}END{print m}' $O/nvsmi.csv)" | tee -a $O/README.txt
echo R4_AUDIO_DONE $TAG
