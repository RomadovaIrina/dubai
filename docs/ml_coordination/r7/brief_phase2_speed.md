# DabAI phase 2 (29.10.2026): SPEED WITHOUT LOSING QUALITY

**Target:** ≤ 300 s per video-minute, end-to-end, for one video on ONE RTX 5090. Audio + face run sequentially on the same GPU.

**Now:** ~480–500 face + ~120–140 audio ≈ 620 s/video-min on a full-face video.
- Face: LatentSync ~370, ESRGAN layer SR ~80, composite ~45–80.

**Quality reference:** the tag `r7-quality-baseline` (m11, with 05 on audio-r7-v3a1). Every speed change must:
1. pass `scripts/r7/r7_baseline.py gate` on all 5 videos (the tolerances are in `quality_baseline.json`);
2. ship a WATCH sbs baseline | new for 05, 04 and 03 for the user's eyes before it becomes the default.

A change that is faster but fails the gate is NOT taken. Report it and move on.

**Rules:**
- Every lever goes behind a flag, measured alone and then stacked.
- Report s/video-min per stage for each lever.
- No downloads over 5 GB without asking. Commercially clean models only (no new model families without asking).
- Push every commit. Send a 5-line status every ~45 min.
- Poll your own jobs: do not wait for background notifications.

## F, the face stage (server F): goal ≤ ~200 s/video-min
Levers, cheapest and safest first:
1. **LatentSync inference:**
   - fewer denoising steps (sweep, e.g. 20 → 10 → 8 → 6);
   - fp16 / bf16 everywhere, the VAE included;
   - the batch size / frames per window;
   - SDPA / flash attention check;
   - `torch.compile`, or CUDA graphs on the UNet;
   - the DeepCache-style reuse of UNet features across steps.
2. **Generate fewer frames:**
   - frames where the dub is silent AND the source mouth is closed / no face → pass-through (already partly done; measure the share);
   - check the 25 fps resampling cost.
3. **Real-ESRGAN layer SR:**
   - a single pass instead of temporal ±1, if the gate holds;
   - fp16, tiling;
   - overlap it with the next chunk's generation.
4. **Composite:**
   - cache the source landmarks / frame map (≈ −20);
   - vectorise fix30;
   - avoid re-encoding twice.
5. **Pipeline overlap:** CPU stages (mediapipe, the composite) run while the GPU generates the next segment.

Deliverable: a table "lever → s/video-min per stage → gate PASS/FAIL", then the best passing stack as an `m12-fast` WATCH.

## A, the audio stage (server A): goal ≤ ~80 s/video-min
1. **Chatterbox:**
   - batch the units;
   - fp16 / bf16;
   - fewer QA retakes when the first take passes;
   - `torch.compile`.
2. **Qwen 14B:**
   - `--dur-rounds 1` (already measured as the same speed as 7B, best faithfulness);
   - fewer candidates;
   - keep the model loaded across stages; KV / prompt cache.
3. **Whisper large-v3:**
   - faster-whisper with batched inference, int8_float16;
   - VAD-segmented batches (the new `--asr-segments vad`).
4. **pyannote:**
   - GPU, batch size;
   - skip it when VAD shows a single speaker? (Only if it is safe; report it.)
5. **Separation, mixing, retake:** measure them and cut the obvious waste. Retake only the units that need it.

Quality gate for audio: vs r7-audio-v3a1 on all 5.
- No new gap junk.
- Faithfulness ≥ baseline − 0.05.
- WER ≤ baseline + 0.01.
- The pause metrics are not worse.
- The clicks are not more.
- The audible AB check stays with the user.

Deliverable: the same table plus an `audio-r7-fast` handoff to F, so the face can be rendered on it and E2E measured.

## End-to-end
At the end, F runs the full one-command E2E on one GPU (audio + face) for all 5 and reports the total s/video-min per video, plus the VRAM peak (spec item 0.4: what stays resident, what is unloaded).
