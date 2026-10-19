# All-GPU audit of the E2E pipeline (round 3, Agent D, RTX 5090 32 GB, 23-CPU quota)

Runner: `run_clean_pipeline_05.py` with the clean defaults: `--audio-policy r2`, Spleeter torch bed, MediaPipe face stack, mouth-only
composite, no CodeFormer. Every model is on the GPU:
- Silero `--vad-device cuda`;
- faster-whisper large-v3 int8_float16;
- pyannote 3.1;
- Qwen2.5-7B Q8_0 through the CUDA llama-cpp-python build, `n_gpu_layers=-1`;
- Chatterbox v3 fp16;
- TTS QA (Whisper + wav2vec2);
- Spleeter torch;
- LatentSync 1.6.

Measured with:
- `DUB_STAGE_LOG`: wall time, model loads, torch peaks, device VRAM at each stage end;
- `nvidia-smi` every 200 ms: the device total, which also sees llama.cpp.

Tools: `scripts/quality/r3_gpu_audit.sh`, `r3_gpu_audit.py`, `r3_gpu_phases.py`. Raw data: `/workspace/dub/artifacts/r3d/audit/<plan>/`.

Speed numbers are the 25 fps runner path. The FIX30 source-fps composite adds about 100 s/video-min of CPU on top; see the last section.

## 1. Plan (a): every model loaded and freed per stage (runner default), Qwen on the GPU

| phase (03, 90.3 s) | device | wall s | s/video-min | peak VRAM | mean VRAM | GPU util |
|---|---|---:|---:|---:|---:|---:|
| VAD + ASR + Qwen segmentation + pyannote + translation | GPU | 43.7 | 29 | 12.0 GiB | 3.9 GiB | 12 % |
| TTS: Chatterbox + QA + Qwen fit / refit | GPU | 124.8 | 83 | 14.7 GiB | 13.4 GiB | 13 % |
| align, Spleeter bed, mix, speech gate | GPU (Spleeter 1.6 s) | 18.6 | 12 | 4.8 GiB | 4.2 GiB | 3 % |
| MediaPipe face stack | **CPU** | 66.8 | 44 | – | – | 0 % |
| LatentSync load + inference + assembly | GPU | 515.2 | **342** | **24.7 GiB** | 21.6 GiB | 79 % |
| **job** | | **769** | **511** | **24.7 GiB** | | |

The same plan on 04 (53.9 s):
- 442 s in total = 492 s/video-min;
- LatentSync 322 s/video-min, at a peak of 22.1 GiB;
- audio phases 122 s/video-min.

### Model loads per video (plan a)

| model | load s | VRAM after load |
|---|---:|---:|
| LatentSync | 19.7–20.0 | +2.7 GiB weights; the working set is reached in inference |
| Chatterbox v3 | 16.0–17.6 | +1.5 GiB (fp16) |
| faster-whisper large-v3 | 8.6–9.8 | ~1.8 GiB (int8_f16) |
| Qwen Q8_0 (CUDA) | 10.5 (translation_load incl. prompt prep), ~2.5 s pure load | 7.7 GiB (nvidia-smi) |
| pyannote 3.1 | 2.1–6.1 | 0.03 GiB weights, 2.7 GiB working |
| TTS QA (Whisper + wav2vec2) | 2.5 | +0.4 GiB alloc |
| Spleeter torch | 0.3 | +0.1 GiB |

All loads together come to 51–55 s per video:
- 36 s/video-min (12 % of the 300 s/min budget) on 03;
- 57 s/video-min (19 %) on 04.

## 2. Unload and baseline check (device VRAM at each stage end, nvidia-smi)

| point (03) | MiB |
|---|---:|
| start | 2 |
| after ASR (Whisper freed) | 660 |
| Qwen loaded (segmentation, translation) | 8 356–8 474 |
| TTS (Chatterbox + QA + Qwen) | ≤ 15 090 |
| **after the Qwen release** (`Llama.close()` + gc) | **2 820** |
| after Spleeter bed / mix | 4 902 (torch reserved cache 4.2 GiB, allocated 48 MiB) |
| LatentSync load | VRAM drops to ~0.9 GiB first (cache freed), then weights |
| end of run | 2 |

The Qwen release works: 8.4 GiB is returned. No stage leaks into LatentSync; the drop before the LatentSync load is visible in the plot. The
remaining reserved cache after the bed is now released explicitly before the video stage (`audio_cache_released` mark). Re-check run:
`audit/gpu` v2; the v1 run is in `audit/gpu_v1`.

![VRAM over time, 03, plan (a)](vram_gpu_03_v1_cachecarry.png)

## 3. Pipeline peak = LatentSync, and it grows with the longest lip-sync segment

| video | longest LATENT_SYNC segment | LatentSync peak (device) |
|---|---|---:|
| 04 | 430 frames (17.2 s) | 22.1 GiB |
| 03 | 1 038 frames (41.5 s) | 24.7 GiB |

The audio stages never exceed 15 GiB. The pipeline peak is always LatentSync. It rises with the segment length because the whole segment's
frames / latents are held on the GPU. A 5-minute customer video with one long continuous speech burst could pass 32 GiB.

**Proposal to A (generator owner):** cap a LatentSync call at ~400 frames by cutting long segments at a 16-frame window boundary, with the
existing continuation overlap, or stream finished frames to the CPU. That bounds the peak at about 22 GiB whatever the video length.

## 4. Residency plans compared (04 then 03 in one worker process)

| plan | what stays on the GPU | 04 wall (s/video-min) | 03 wall (s/video-min) | pipeline peak | result |
|---|---|---:|---:|---:|---|
| (a) load / unload per stage | nothing between stages | 442 (492) | 769 (511) | 24.7 GiB | PASS |
| (b) audio models resident, Qwen released before LatentSync | Silero, Whisper, pyannote, Chatterbox, TTS QA, LatentSync | 436 (485) | **694 (461)** | **30.4 GiB** | PASS, 1.4 GiB headroom |
| all resident incl. Qwen (`--qwen-resident on`) | everything | 485 (540) | 326: **OOM** | 31.4 GiB (full) | **FAIL**: 03 LatentSync OOM in 3 segments → pass-through, validation FAIL |
| (c) Qwen partial offload | not run | – | – | – | not needed: Qwen is 7.7 GiB, loads in ~2.5 s and is released before LatentSync anyway |

Notes on the plans:
- **(b) vs (a):**
  - on a warm job (03), (b) saves the reloads (Whisper 9 s, pyannote 2–6 s, Chatterbox 17 s, LatentSync 20 s): −76 s = **−50 s/video-min**;
  - the first job pays the same loads as (a);
  - with 8–10 GiB of audio models resident, the LatentSync peak on 03 (1 038-frame segment) was 30.4 GiB, so (b) is safe only with the
    segment cap from §3 (22 + 9 ≈ 31 GiB worst case today).
- **All resident does not fit 32 GB:** Qwen 7.7 + audio models ~9 + LatentSync 21–25 GiB.

**Recommendation:**
- Use **(b) with the LatentSync segment cap**: audio models plus LatentSync resident; Qwen loaded per job (2.5 s) and closed before LatentSync.
- Without the cap, use (a) on 32 GB cards.

## 5. Qwen on the GPU vs the CPU (04, the same run otherwise)

| | Qwen CPU (frozen baseline) | Qwen GPU |
|---|---:|---:|
| segmentation s | 47.3 | 0.9 |
| translation s | 35.6 | 0.6 |
| TTS stage s (includes Qwen fit / refit) | 374.3 | 51.0 |
| audio phases s/video-min | 561 | 122 |
| job s/video-min | 942 | 492 |
| peak VRAM | 21.1 GiB (LatentSync) | 22.1 GiB (LatentSync) |

Qwen on the GPU is required for the norm.

Output equivalence (C, 04 / 01 / 03, same build):
- the segmentation is identical;
- translation is identical on 37 of 41 units, the rest differ in wording only;
- Q5_K_M drifts (CJK characters), so keep Q8_0.

## 6. Minimum GPU memory, two jobs per card, and the 300 s/min norm

- **Minimum card:** the peak is 22.1 GiB (04) to 24.7 GiB (03) with plan (a).
  - It does **not** fit a 24 GB card today: 03 needs 24.7 GiB of the 23.6 GiB usable.
  - With the segment cap (~22 GiB) it fits 24 GB with ~1.5 GiB headroom.
  - A 32 GB card has 7 GiB headroom with plan (a), ~1.5 GiB with plan (b).
- **Two jobs per 32 GB card:** no. Two LatentSync working sets (2 × 20–25 GiB) do not fit.
  - The audio stage of job N+1 (≤ 15 GiB) could overlap the LatentSync of job N only on a 48 GB card.
  - On 32 GB the useful overlap is the CPU work: the face stack (44 s/video-min, CPU) next to the audio stage of the same job, and FIX30 of
    job N next to the GPU stages of job N+1.
- **Norm 300 s/video-min:**
  - the GPU path is 461–511 s/video-min at 25 fps, plus FIX30 (~100 s/video-min CPU);
  - **LatentSync alone is 317–342 s/video-min** and over the norm by itself (20 steps, DeepCache 5, window batch 2);
  - the audio stage is now 122 s/video-min, and the face stack 44 s/video-min on the CPU (can run during the audio stage).

## 7. Top costs and cheap wins (no model-setting changes)

| # | cost (s/video-min) | cheap win | expected |
|---|---:|---|---|
| 1 | LatentSync 317–342 | model setting changes are A's: propose step / DeepCache / batch A/B with a human check; plan (b) saves the 20 s load per job | −13–20 s/min (load) |
| 2 | TTS stage 57–83 | plan (b): Chatterbox + QA resident (−17 s load / job); Qwen GPU already done | −11–19 s/min |
| 3 | MediaPipe face stack 44 (CPU) + FIX30 ~100 (CPU) | run the face-stack prep in parallel with the audio stage (it only needs the master; `--prep-only` exists); FIX30 threads ≤ 8 and overlapped with the next job's GPU stages | −44 s/min off the critical path; FIX30 off the path in a worker |

Measured E2E of the r3F-like candidate on 04 (`r3_e2e.sh`, Qwen still on the CPU in that script) was 1 132 s/video-min. With Qwen on the GPU
and the wins above, the critical path is about **LatentSync 320 + audio 120 + glue ≈ 450–480 s/video-min**. That is still above 300 until
LatentSync gets faster: steps / DeepCache are A's decision and need a human visual check.
