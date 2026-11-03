# 0.4 VRAM of the current staged pipeline (r13, RTX 5090 32 GB)

Measured on the one-command E2E (`scripts/r9/e2e.sh`, defaults + `E2E_SYNC=1`), r12 lineage, 03.11. Device memory = nvidia-smi, 100 ms
samples per stage (`work/nvsmi.csv`); per-model numbers = the pipeline's own stage log (`DUB_STAGE_LOG`, 04, job f_oom).

## Per stage (device MiB)

| Stage | old03 (90 s, 464×848) peak / p50 | m18 (20 s, 720×1280) peak / p50 | m03 (40 s, 720×1280) peak / p50 | What is on the GPU |
|---|---|---|---|---|
| precheck | 18 | 18 | 18 | ffprobe + MediaPipe (CPU) |
| **audio** | **30 180** / 26 666 | 26 478 / 21 180 | **30 410** / 25 952 | ONE process: Whisper large-v3 (CT2 int8_float16) → pyannote 3.1 → Qwen 2.5 14B Q5_K_M (llama.cpp, all layers) + batched Chatterbox v3 + TTS QA (wav2vec2) + Spleeter; **plus** htdemucs in a second process (R9_SEP_OVERLAP) |
| sep | 2 | 2 | 2 | (the demucs mix was already made during audio) |
| face_gen | 13 750 / 13 744 | 13 134 / 12 848 | 13 080 / 13 080 | LatentSync 1.6 (fp16, DeepCache, VAE compile) + MediaPipe crops |
| face_comp | 7 441 / ~0 | 13 113 / ~0 | 10 233 / ~0 | CodeFormer detail + layer SR bursts; the rest of the composite is CPU |
| subs / mux | 2 | 2 | 2 | CPU (libass, x264) |

## Per model inside the audio process (04, stage log)

| Step | Device used after (MiB) | Torch peak (MiB) | Note |
|---|---|---|---|
| Whisper large-v3 ASR | 528 | – | CT2 frees most of it after the call |
| pyannote 3.1 diarization | 722 | 2 722 | |
| **Qwen 14B load** | **11 372** | – | +10.7 GB (weights + 4k context) |
| Chatterbox v3 + TTS QA load | – | +1 529 / +360 | |
| TTS (batched) until Qwen released | **25 578** | 8 365 | Qwen + Chatterbox activations + allocator cache |
| Spleeter bed | 26 462 | 8 413 | |

## Answer to 0.4

- **Everything resident does NOT fit:** audio peak 30.4 GB + LatentSync 13–14 GB ≈ 44 GB > 32 GB.
- **The current staged design stays:** the audio process (Qwen + TTS) is released before the face generator, and the composite runs after it. Same conclusion as the 28.10 status; the numbers have moved:
  - audio 22–25 → 26–30 GB (14B on the GPU + batched TTS + the htdemucs overlap);
  - face_gen ~14 GB.
- **Headroom is thin in the audio stage:** 30.4 of 32.6 GB.
  - Anything else on the card at the same time (a second job's composite) risks OOM. That is what the r10 worker's admission rule (R10_HEAVY_LOCK / R10_COMP_GATE) protects.
  - Also: the audio stage does NOT take `gpu.lock` for its whole duration, so ad-hoc jobs "behind the lock" are not safe next to it.
- **What could share the card if needed:** face_gen (13–14 GB) plus a composite (≤ 13 GB) of another job, ≈ 27 GB, which is what the r10 worker overlaps.
