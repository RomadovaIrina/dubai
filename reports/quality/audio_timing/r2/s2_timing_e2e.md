# r2 step 2: the timing policy in the real pipeline (01–05, RTX 5090, 2026-10-17)

**Runs.** `run_clean_pipeline_05.py --alignment burst --stop-after audio --qwen-gpu-layers -1`, once with `--audio-policy cand5`
(yesterday's reference flags) and once with the new default `--audio-policy r2` (= cand5 + `--sentence-flow-gap 0.2 --spread pauses
--fill-slowdown 0.875`). Bed: Spleeter (does not affect these dub-track metrics). Work dirs: `/workspace/dub/runs/e2e_{cand5,r2}/`.
r2 and cand5 share **every** translation and raw Chatterbox take (01–05: 66 of 66 units byte-identical), so the table below is a controlled
A/B of the timing layer inside the real pipeline. Today's cand5 vs the canonical baseline: 04 6/6 units identical, 03 8/10 (u01 / u02 fit-up
wording differs: GPU vs CPU Qwen); 01 / 05 canonical not yet on this server (05's cand5 gaps are smaller today: 5.5 vs 7.2 s, so its
translation / fit-up differ from yesterday's).
**LS mask.** No video run here: `face_ref.py` = MediaPipe router classes × speech gate of the cand5 dub track (production video-stage rule).
It reproduces yesterday's LS-frame counts from the real LatentSync runs exactly on 03 (240 / 143) and 04 (106 / 67).

## Gaps "source speaks, dub silent" (Silero both sides, dub VAD ≥ 60 ms), weak-boundary pauses, tempo

| video | policy | source w/o dub s | gaps ≥ 0.35 s n / s / LS frames | > 0.5 s n / s | > 1.0 s n / s | longest s | weak-boundary pauses | parts < x0.9 | yesterday offline (same policy) |
|---|---|---:|---|---|---|---:|---:|---:|---|
| 01 | cand5 | 8.2 | 8 / 5.5 / 15 | 4 / 3.7 | 1 / 1.7 | 1.7 | 1 | 0 % | 8.4 · 8 / 5.3 / 15 · 4 / 3.6 · 1 / 1.7 |
| 01 | **r2** | 7.6 | **5 / 3.2 / 13** | 3 / 2.3 | 1 / 1.1 | 1.1 | 0 | 37 % | 7.7 · 5 / 3.0 / 13 · 2 / 1.7 · 1 / 1.1 |
| 02 | cand5 | 11.3 | 13 / 8.0 / 0 (no face) | 7 / 5.2 | 0 | 0.9 | 1 | 0 % | (not measured yesterday) |
| 02 | **r2** | 11.2 | **12 / 7.0 / 0** | 5 / 3.9 | 0 | 0.9 | 1 | 37 % | |
| 03 | cand5 | 11.1 | 12 / 9.4 / 240 | 6 / 6.7 | 3 / 4.3 | 2.0 | 2 | 0 % | 11.1 · 12 / 9.4 / 240 · 6 / 6.7 · 3 / 4.3 (identical) |
| 03 | **r2** | 9.7 | **8 / 5.6 / 143** | 4 / 3.8 | 1 / 1.3 | 1.3 | 1 | 60 % | 9.5 · 8 / 5.6 / 143 · 4 / 3.8 · 1 / 1.3 (identical) |
| 04 | cand5 | 6.1 | 6 / 4.1 / 106 | 4 / 3.3 | 0 | 1.0 | 1 | 0 % | identical |
| 04 | **r2** | 5.6 | **5 / 2.6 / 67** | 3 / 1.8 | 0 | 0.6 | 1 | 58 % | identical |
| 05 | cand5 | 5.5 | 6 / 4.6 / 88 | 5 / 4.2 | 1 / 1.3 | 1.3 | 0 | 0 % | 7.2 · 8 / 6.4 / 164 (other cand5 translation) |
| 05 | **r2** | 5.3 | **5 / 3.5 / 58** | 5 / 3.5 | 1 / 1.1 | 1.1 | 0 | 62 % | 6.9 · 7 / 4.8 / 121 |
| **sum 01+03+04+05** | cand5 → r2 | 30.9 → 28.2 | 32 / 23.6 / 449 → **23 / 14.9 / 281** (−37 % LS frames) | 19 → 15 | 5 / 7.3 → 3 / 3.5 | 2.0 → 1.3 | 4 → 2 | | yesterday 525 → 344 (−34 %) |

Tempo: speed-up max unchanged (x1.15, no part above the cap); slow-down min x0.875 on 37–62 % of the parts (as offline).

## Intelligibility (CPU Whisper int8 WER / wav2vec2 CTC CER, `eval_cpu.py`; bad = CTC CER ≥ 0.3)

| video | cand5 WER / CER / bad | r2 WER / CER / bad | note |
|---|---|---|---|
| 01 | 0.027 / 0.050 / 0 | 0.027 / 0.052 / 0 | |
| 02 | 0.095 / 0.157 / 3 | 0.102 / 0.159 / 3 | the +0.007 is one unit (u03 "chaise longues": the same mispronounced take, Whisper hears "she's lungs" → "sheaths and some longs" when slowed) |
| 03 | 0.004 / 0.046 / 0 | 0.004 / 0.038 / 0 | |
| 04 | 0.009 / 0.048 / 0 | 0.009 / 0.051 / 0 | |
| 05 | 0.012 / 0.144 / 0 | 0.000 / 0.163 / 0 | CTC CER +0.019 = yesterday's offline sign of slower speech at 0.875 (+0.02) |

No interjections, no missing final words in any run. Old gains kept: weak-boundary pauses 4 → 2 (01+03+04+05), the 03 24.5 s window
(dub covers it, gap < 0.35 s), atempo cap respected, no unit became bad.

## Audio-stage runtime (s per video-minute, r2 runs, Qwen + TTS + QA on the GPU, Spleeter on the CPU)

| video | duration s | audio stage wall s (runner `total`) | s / video-min |
|---|---:|---:|---:|
| 01 | 101.7 | 79.4 (TTS 39.5, ASR 7.0, diarization 5.5, Spleeter 7.3) | 46.9 |
| 02 | 110.9 | 65.8 | 35.6 |
| 03 | 90.3 | 77.0 | 51.2 |
| 04 | 53.9 | 43.2 | 48.1 |
| 05 | 52.4 | 52.9 | 60.7 |

Audio stage = 36–61 s per video-minute (incl. model loads, cold per run), i.e. ≤ 20 % of the 300 s/min budget for the whole pipeline.
Qwen on the GPU (CUDA llama.cpp, `scripts/setup_llama_cuda.sh`): translation 0.6–2.2 s per video.

The two reviewed 03 windows: 24.5–25.0 s gap 0.5 → **0.1 s**; 34.3–35.5 s 1.2 → 0.9 s and 37.5–38.5 s 1.0 → 0.5 s (u04: translation length,
layer 3 / video side).
