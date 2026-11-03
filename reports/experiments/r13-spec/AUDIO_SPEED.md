# 0.5 audio-stage speed: profile and output-preserving levers (r13, 03.11)

**Setup:**
- The e2e audio + sep stages (lip-sync off) on old03 (90 s), m03 (40 s), m18 (20 s) = 2.515 video-min.
- Current defaults: r9fast-audio, approved.
- GPU-busy = the NVML sampler (`scripts/r10/gpu_sampler.py`, 100 ms, device SM util × dt). Each sample goes to the innermost open sub-step of the `R9_TIMELINE` timeline (`scripts/r13/prof_audio.py`, the `scripts/r9/gpu_busy.py` rule).
- Equivalence: the pipeline is **deterministic**. Two plain baseline runs give a bit-identical voice track and a Spleeter mix equal to ~120 dB SNR (only the htdemucs bed varies: 34–47 dB). So a lever counts as output-preserving only if the transcript, the translations, the placed texts and the voice track are bit-identical (`scripts/r13/cmp_runs.py`).

## Where the audio GPU time goes (baseline)

| sub-step | GPU-busy s/vmin | wall s/vmin |
|---|---:|---:|
| Qwen 14B (batched resident engine: duration-aware candidates, judge, retake) | **15.5** | 18.1 |
| TTS QA (Whisper large-v3 int8, beam 5, every take) | **8.3** | 13.2 |
| S3Gen vocoder (batched CFM) | 6.6 | 8.7 |
| T3 decode (batched) | 6.3 | 9.6 |
| other: process start, loads between stages, CPU chain | 2.6 | 21.3 |
| ASR (Whisper large-v3) | 1.2 | 6.0 |
| chain: lip-placement Whisper | 0.8 | 1.9 |
| diarization, translation load, segmentation, the rest | < 1 each | |
| **audio + sep total** | **43.6** (device-busy count: 54–62) | 97 |

Alongside, not exclusive: MediaPipe source track (CPU), htdemucs stems 0.8, demucs remix 0.2.

**In the 300 s/vmin budget:** the audio stage is ~44 weighted (54–62 by the "any kernel running" count the e2e reports), against ~280 for the face. Qwen + QA + Chatterbox are 37 of the 44.

## Levers tested (same 3 videos)

| lever | flag | GPU-busy total s/vmin (sub-step) | outputs | verdict |
|---|---|---|---|---|
| baseline | – | 43.6 (repeat 43.1) | – | reference |
| S3Gen CFM estimator `torch.compile` (dynamic) | `R13_S3_COMPILE=1` | 46.4; S3Gen wall 8.5 → **22.9** (a recompile per take length) | different takes chosen (numerics flip QA decisions) | **rejected** |
| QA Whisper beam 5 → 1 | `DUB_QA_BEAM=1` (existing) | 42.8 (QA 8.6 → 7.4) | **bit-identical** voice on all 3 | candidate: −1.2 QA, about −0.3 to −1 total; needs a wider bit-identity check (01–05) before a default change |
| QA beam 1 + Whisper float16 | `+ DUB_QA_COMPUTE=float16` (new, default int8_float16) | 41.4 (QA 6.4) | bit-identical on old03 | **rejected**: m03 audio stage at 31.95 GB of 32 GB → failed (fp16 Whisper needs more VRAM than int8 next to Qwen + Chatterbox) |

Not re-tried (in the r9 audio report, and not output-preserving or no gain):
- batched QA (`DUB_QA_BATCH`): about 0 gain, and OOM inside the full stack;
- more Qwen batching: already the resident batched engine, flash attention, prefix cache, 2048 batch;
- batched Chatterbox: already the default (`DUB_TTS_BATCH`, `DUB_S3_BATCH`).

Resident models across videos (the r9 audio worker, `R9_WORKER_SPOOL`) saves load WALL time (~15–20 s per job). The loads are not GPU-busy (the "other" row: 2.6 s/vmin).

## Conclusion
- **No output-preserving lever left that moves the audio stage materially.** The best clean candidate (QA beam 1) is ~1 s/vmin.
- The audio stage is already ~15 % of the budget. **The ≤ 300 norm has to come from the face**, e.g. the LS agent's `LS_RATE=2/3`: face 302 → 215 s/vmin at equal SyncNet on m18.
- The levers that would cut more audio GPU time change the output and need an ear AB:
  - fewer judge / candidate Qwen calls;
  - QA only on borderline takes, or a smaller QA model.
