# audio-r3 (Agent C round 3, 19.10, server A): same layout as audio-r2

Files per video: `<id>_r3.manifest.json`, `<id>_r3.residual.json`, `<id>_r3.gaps.json`, plus `dubbed_16k.wav` (LatentSync input + speech gate) and `final_mix_aac.m4a` / `final_mix_44k.wav` (audio to mux). Usage as before: `candidate_video_stage.py --dubbed-dir <this>/<id>`.

**What is new vs audio-r2.** The timing policy is unchanged (r2: flow 0.2 + spread + slow-down floor 0.875, Spleeter bed, residue duck, duck −4 dB / 250 ms / 600 ms).
- **Numbers and units are spoken words before TTS.** The text is normalised before synthesis, so QA, alignment and evaluation all see the same words. 01 "tile 600x600 … 9 mm": audio-r2 was heard as "600 x 67"; audio-r3 is heard as "600 by 600 … 9 millimeters".
- **Qwen drift guard.** Foreign-script characters or bracketed asides in a translation trigger a retry or get stripped; such rewrites are rejected. It did not trigger on 01–05.
- **Everything runs on the GPU.**
  - Qwen Q8_0 runs on the GPU: all layers for segmentation and translation, 14 layers during TTS. Segmentation is identical to CPU; translations differ in wording on a few units.
  - The Spleeter bed comes from a PyTorch port of the same MIT weights, identical to TF at 118 dB SNR.
  - The TTS QA uses Whisper int8_float16.
- **Declick now iterates and never raises a peak.** 02 96.17 s and 04 22.99 s: 0 clicks left.
- Speed: 151–227 s per video-minute (audio-r2 on CPU: 478–632). Peak VRAM 9.2–9.3 GB.
- **Not included yet:**
  - loudness normalisation (both rounds are −12.9…−20.4 LUFS with true peak up to −0.2 dBTP; next update);
  - the layer-3 refit (it will be run on top of audio-r3, not r2).

## Intelligibility: Whisper large-v3 WER / wav2vec2 CTC CER per unit (CPU eval, the same tool for both rounds)

| video | audio-r2 WER / CER | audio-r3 WER / CER | note |
|---|---|---|---|
| 01 | 0.028 / 0.057 | **0.017** / 0.068 | the number sentences are fixed |
| 02 | 0.060 / 0.164 | 0.075 / **0.140** | "Hamam" is heard as "Our mom" (rare word, onset lost), worse than r2's "Tamam" |
| 03 | 0.008 / 0.038 | 0.010 / 0.053 | different GPU-Qwen wording |
| 04 | 0.009 / 0.037 | 0.009 / 0.049 | |
| 05 | 0.011 / 0.149 | 0.011 / **0.141** | |

The worst 20 units of each round are in C's outbox as `listen/audio-r2_worst*.wav` and `listen/audio-r3_worst*.wav`. `listen/README.md` lists the expected text, what Whisper heard and what CTC heard.

## Extraneous sounds (audio_stages.py detectors on the delivered tracks)

| video | clicks in dub voice r2 → r3 | mix-only clicks | hot cut edges | extra voice in TTS | Russian voice in bed (s) | bed steps created by the mix |
|---|---|---|---|---|---|---|
| 01 | 0 → 0 | 0 → 1 | 0 → 0 | 3 → 3 | 0 → 0 | 4 → 4 |
| 02 | 0 → 0 | 1 → 2 | 4 → 4 | 2 → 1 | 0 → 0 | 9 → 11 |
| 03 | 0 → 0 | 0 → 1 | 1 → 1 | 0 → 0 | 0 → 0 | 2 → 2 |
| 04 | 0 → 0 | 0 → 0 | 1 → 1 | 0 → 1 | 0 → 0 | 0 → 1 |
| 05 | 0 → 0 | 0 → 0 | 2 → 3 | 1 → 2 | 0 → 0 | 3 → 3 |

## Residual gaps "speaker talks, dub silent" (≥ 0.35 s; LS frames = MediaPipe router × speech gate of this dub)

| video | audio-r2 n / s / LS frames | audio-r3 n / s / LS frames |
|---|---|---|
| 01 | 7 / 3.8 / 13 | 6 / 3.7 / 11 |
| 02 | 8 / 4.3 / 0 | 9 / 5.0 / 0 |
| 03 | 7 / 4.6 / 118 | 8 / 5.2 / 133 |
| 04 | 5 / 2.6 / 67 | 5 / 2.3 / 59 |
| 05 | 4 / 2.2 / 43 | 4 / 2.2 / 43 |

## Loudness of `final_mix` (ffmpeg ebur128), not yet normalised

| video | audio-r2 LUFS / true peak dBTP | audio-r3 LUFS / true peak dBTP |
|---|---|---|
| 01 | −12.9 / −0.2 | −13.4 / −0.2 |
| 02 | −20.0 / −1.4 | −20.4 / −1.4 |
| 03 | −17.5 / −0.2 | −17.1 / −0.2 |
| 04 | −13.5 / −0.2 | −13.7 / −0.2 |
| 05 | −18.9 / −3.3 | −18.5 / −3.2 |

## Processing time and VRAM (audio-only runner, shared RTX 5090)

| video | s | s per video-minute | peak VRAM GB |
|---|---:|---:|---:|
| 01 | 340 | 201 | 9.3 |
| 02 | 280 | 151 | 9.3 |
| 03 | 249 | 165 | 9.3 |
| 04 | 189 | 210 | 9.3 |
| 05 | 198 | 227 | 9.2 |

Breakdown: Chatterbox generation 70–117 s; Qwen fit-up rewrites on 14 GPU layers 84–115 s (13 s with all layers on the GPU). Next speed step: Qwen and Chatterbox in alternating phases, so Qwen runs with all layers on the GPU within 10 GB.

**Verdict for the video agents.**
- **Lip-sync:** the residual gaps are about the same as audio-r2, so lip-sync work transfers.
- **Take audio-r3 for:**
  - number / unit sentences (01);
  - the GPU speed.
- **Watch:** 02's rare word "Hamam" is less clear in this round; the fix is in progress.
