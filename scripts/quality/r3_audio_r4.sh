#!/bin/bash
# Round 3 D: audio-r4 = audio-r3 settings (r2 policy, speakable text, lexicon, Qwen + Spleeter + QA on the GPU) + the phrase-cut fix
# (DUB_CUT_SNAP=1: inner cuts at the energy minimum past the early CTC word end; DUB_EQPOWER_FADE=1: equal-power fades, inner part edges
# >= 12-30 ms, unit edges 10-12 ms) + loudness normalisation (-16 LUFS, TP -1.5 -> <= -1 dBTP after AAC). Checks: hard cuts, levels.
#   bash scripts/quality/r3_audio_r4.sh [ids]      -> /workspace/dub/artifacts/r3d/audio-r4/{runs/,<id> -> work dir}
set -u
DUB_ROOT=/workspace/dub/dubai; export DUB_ROOT; . $DUB_ROOT/scripts/env.sh >/dev/null; cd $DUB_ROOT
R=/workspace/dub/artifacts/r3d/audio-r4; mkdir -p $R/runs; export PYTHONPATH=third_party/llama_cuda${PYTHONPATH:+:$PYTHONPATH} PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export DUB_CUT_SNAP=1 DUB_EQPOWER_FADE=1
for id in ${*:-03 04 05 01 02}; do
  src=$(ls /workspace/dub/inputs/test_videos_raw/$id.* | head -1); s=$(date +%s)
  flock /workspace/dub/gpu.lock python scripts/pilot/run_clean_pipeline_05.py --input $src --output $R/runs/${id}_r4.mp4 --target-lang en --alignment burst \
    --stop-after audio --codeformer off --face-stack mediapipe --audio-policy r2 --qwen-gpu-layers -1 --tts-text-norm on --qa-whisper-compute int8_float16 \
    --work-dir $R/runs > $R/runs/${id}.log 2>&1 || { echo "FAIL audio $id"; continue; }
  W=$R/runs/work_$(basename ${src%.*}); ln -sfn $W $R/$id; cp $R/runs/${id}_r4.manifest.json $W/${id}_r4.manifest.json
  python scripts/quality/loudness.py --dir $W > $R/runs/${id}.loud.log 2>&1 || echo "WARN loudness $id"
  echo "== $id audio-r4 $(( $(date +%s) - s )) s; $(python scripts/quality/r3_hardcut.py $W/dubbed_24k.wav | sed 's|.*: ||')"
done
echo AUDIO_R4_DONE
