#!/bin/bash
# audio-r3 candidate a: Q8 all GPU layers (segmentation + translation) -> 14 layers in the TTS session, int8_float16 QA, speakable text, drift guard.
# Sequential (VRAM <= ~10 GB). VRAM sampler bound to the runner PID.
cd /workspace/dub/dubai-audio
for id in ${IDS:-03 01 04 05 02}; do
  scripts/quality/r3_audio_run.sh $id r3a --qwen-quant q8_0 --qwen-tts-gpu-layers 14 --qa-whisper-compute int8_float16 --tts-text-norm on > /workspace/dub/c/logs/run_${id}_r3a.out 2>&1 &
  R=$!
  while kill -0 $R 2>/dev/null; do
    nvidia-smi --query-compute-apps=pid,used_memory --format=csv,noheader,nounits | while IFS=, read p m; do case "$(ps -o args= -p $p 2>/dev/null)" in *run_clean_pipeline_05*r3a*) echo "$(date +%T) $id $m";; esac; done
    sleep 1
  done >> /workspace/dub/c/logs/vram_r3a.log
done
echo ALL_DONE >> /workspace/dub/c/logs/vram_r3a.log
