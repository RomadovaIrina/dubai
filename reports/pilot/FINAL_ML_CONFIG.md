# DabAI pilot — frozen final ML configuration (2026-09-18)

One configuration for every measurement from pilot 0.5 onward (0.5 re-measurement, final outputs, 0.8). Not changed
between videos. Environment: `/venv/dabai`, Python 3.11, PyTorch 2.7.1+cu128, CUDA 12.8, RTX 5090 / sm_120
(`reports/pilot/env/20260918_141354-ed111920.md`).

## AUDIO
| stage | component | setting |
|---|---|---|
| VAD | Silero VAD | default `get_speech_timestamps` |
| ASR | faster-whisper large-v3 | `int8_float16` (executes fp16 on sm_120), word timestamps, beam 5 |
| diarization | pyannote/speaker-diarization-3.1 | GPU |
| translation | Qwen2.5-7B-Instruct **Q8_0** via llama.cpp | CPU, `n_gpu_layers=0`, `DUB_THREADS` threads, pilot 0.9 system prompt, temperature 0 (pilot 0.9 decision: Q8_0 over the 1.48x faster Q5_K_M for translation quality) |
| TTS | Chatterbox Multilingual v3 | fp16 (fp32 fallback), speaker reference = diarized source speech |
| alignment | current duration alignment | ffmpeg atempo speed-up to slot / natural speed + silence padding; timeline never moves |

## VIDEO (`run_clean_pipeline_05.py --video-backend optimized`)
| item | setting |
|---|---|
| router | facexlib RetinaFace ResNet50 fp16, min_w 50, min_h 80, confidence 0.8, aspect 0.2–1.5 |
| router stride | 3 (every 3rd gated frame detected, others inherit) |
| speech gate | ON: VAD(original) ∪ VAD(dubbed track), margin 0.24 s, gaps < 0.6 s merged; outside → NO_SPEECH pass-through |
| boundary crossfade | 4 frames (LATENT_SYNC ↔ pass-through) |
| segments | in-memory (no per-segment mp4 round trips) |
| detection | LatentSync FaceDetector once per frame (verify pass), replayed inside LatentSync; LatentSync safety check retained ("Face not detected" → split / pass-through) |
| DeepCache | ON, cache_interval 5, branch 0 (A/B `reports/pilot/optim/deepcache_interval_ab_04.md`) |
| temporal window batch | 2 |
| attention | native PyTorch SDPA (auto → fused flash kernel) |
| torch.compile / Inductor | OFF |
| TensorRT | OFF |
| CodeFormer | OFF (`--no-codeformer`, never invoked) |
| output | 25 fps CFR master timeline, source resolution, dubbed AAC track over the whole timeline |

## LatentSync (frozen)
stage2_512 · 20 DDIM steps · guidance 1.5 · fp16 · seed 1247 · num_frames 16

## Canonical command (used by the formal 0.5 harness)
```bash
source scripts/env.sh
python scripts/pilot/run_clean_pipeline_05.py --input {input} --output {output} --target-lang en --no-codeformer \
  --video-backend optimized --face-router retinaface --face-min-w 50 --face-min-h 80 --retina-conf 0.8 --router-stride 3 \
  --speech-gate on --gate-margin-s 0.24 --gate-merge-gap-s 0.6 --crossfade-frames 4 --deepcache-interval 5 \
  --window-batch-size 2 --sdpa-backend auto --compile-backend none --seed 1247
```
All of these values are also the runner defaults (`OPTIMIZED_DEFAULTS`); they are passed explicitly for the record.
